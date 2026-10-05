use std::{
    cmp::Ordering,
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::{
    Tui,
    cache::{self, ContextVramEstimate, LocalModelMetadata, ModelInfo},
    gpu::{self, GpuAllocationEstimate, GpuCountEstimate, GpuDetection},
    hub::{self, HubClient, ModelMetadata},
    ui,
};

const EVENT_WAIT: Duration = Duration::from_millis(250);
pub(crate) const VRAM_OVERHEAD_PERCENT: u32 = 20;
const DEFAULT_CONTEXT_TOKENS: u64 = 2_048;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum HubModelState {
    Loading,
    Ready(ModelMetadata),
    Error(String),
}

struct HubJob {
    generation: u64,
    model_ids: Vec<String>,
}

struct HubResult {
    generation: u64,
    model_id: String,
    result: Result<ModelMetadata, hub::HubError>,
}

struct HubWorker {
    pending: Arc<(Mutex<Option<HubJob>>, Condvar)>,
    results: Receiver<HubResult>,
    latest_generation: Arc<AtomicU64>,
}

impl HubWorker {
    fn start() -> std::io::Result<Self> {
        let pending = Arc::new((Mutex::new(None::<HubJob>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let (results_tx, results_rx) = mpsc::sync_channel(64);
        let latest_generation = Arc::new(AtomicU64::new(0));
        let worker_generation = Arc::clone(&latest_generation);
        thread::Builder::new()
            .name("hugtop-hub".into())
            .spawn(move || {
                let client = HubClient::new();
                loop {
                    let job = {
                        let (lock, ready) = &*worker_pending;
                        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
                        while pending.is_none() {
                            pending = ready
                                .wait(pending)
                                .unwrap_or_else(|error| error.into_inner());
                        }
                        pending.take().expect("pending Hub job checked")
                    };
                    if job.generation != worker_generation.load(AtomicOrdering::Acquire) {
                        continue;
                    }
                    for model_id in job.model_ids {
                        if job.generation != worker_generation.load(AtomicOrdering::Acquire) {
                            break;
                        }
                        let result = client.fetch_model(&model_id);
                        if job.generation != worker_generation.load(AtomicOrdering::Acquire) {
                            break;
                        }
                        if results_tx
                            .send(HubResult {
                                generation: job.generation,
                                model_id,
                                result,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            })?;
        Ok(Self {
            pending,
            results: results_rx,
            latest_generation,
        })
    }

    fn replace_pending(&self, job: HubJob) {
        let (lock, ready) = &*self.pending;
        let mut pending = lock.lock().unwrap_or_else(|error| error.into_inner());
        *pending = Some(job);
        ready.notify_one();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SortMode {
    Name,
    Size,
    Modified,
}

impl SortMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Size => "size",
            Self::Modified => "recent",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Size => Self::Modified,
            Self::Modified => Self::Name,
            Self::Name => Self::Size,
        }
    }

    pub(crate) fn direction(self, reversed: bool) -> &'static str {
        match (self, reversed) {
            (Self::Name, false) | (Self::Size | Self::Modified, true) => "ASC",
            (Self::Name, true) | (Self::Size | Self::Modified, false) => "DESC",
        }
    }
}

pub(crate) struct App {
    pub(crate) cache_root: PathBuf,
    pub(crate) models: Vec<ModelInfo>,
    pub(crate) visible: Vec<usize>,
    pub(crate) selected: usize,
    pub(crate) filter: String,
    pub(crate) filter_draft: String,
    pub(crate) editing_filter: bool,
    pub(crate) show_help: bool,
    pub(crate) sort: SortMode,
    pub(crate) sort_reversed: bool,
    pub(crate) gpu_detection: GpuDetection,
    pub(crate) scan_error: Option<String>,
    pub(crate) action_status: Option<ActionStatus>,
    pub(crate) pending_delete: Option<PendingDelete>,
    pub(crate) deleting: Option<PendingDelete>,
    pub(crate) context_tokens: u64,
    pub(crate) online: bool,
    pub(crate) hub_generation: u64,
    pub(crate) hub_states: HashMap<String, HubModelState>,
    pub(crate) local_metadata: HashMap<String, LocalModelMetadata>,
    hub_worker: Option<HubWorker>,
    hub_worker_error: Option<String>,
    should_quit: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PendingDelete {
    pub(crate) id: String,
    pub(crate) path: PathBuf,
    pub(crate) size_bytes: u64,
    model: ModelInfo,
    visible_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ActionStatus {
    Success(String),
    Failure(String),
}

impl ActionStatus {
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Success(message) | Self::Failure(message) => message,
        }
    }
}

#[derive(Debug)]
enum AppAction {
    Refresh,
    Delete(Box<PendingDelete>),
    DrainInput,
}

impl App {
    #[cfg(test)]
    pub(crate) fn new(cache_root: PathBuf, gpu_detection: GpuDetection) -> Self {
        Self::new_with_online(cache_root, gpu_detection, false)
    }

    pub(crate) fn new_with_online(
        cache_root: PathBuf,
        gpu_detection: GpuDetection,
        online: bool,
    ) -> Self {
        let (hub_worker, hub_worker_error) = if online {
            match HubWorker::start() {
                Ok(worker) => (Some(worker), None),
                Err(error) => (
                    None,
                    Some(format!("could not start Hub background worker: {error}")),
                ),
            }
        } else {
            (None, None)
        };
        Self {
            cache_root,
            models: Vec::new(),
            visible: Vec::new(),
            selected: 0,
            filter: String::new(),
            filter_draft: String::new(),
            editing_filter: false,
            show_help: false,
            sort: SortMode::Size,
            sort_reversed: false,
            gpu_detection,
            scan_error: None,
            action_status: None,
            pending_delete: None,
            deleting: None,
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            online,
            hub_generation: 0,
            hub_states: HashMap::new(),
            local_metadata: HashMap::new(),
            hub_worker,
            hub_worker_error,
            should_quit: false,
        }
    }

    pub(crate) fn selected_model(&self) -> Option<&ModelInfo> {
        self.visible
            .get(self.selected)
            .and_then(|index| self.models.get(*index))
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.models
            .iter()
            .fold(0_u64, |total, model| total.saturating_add(model.size_bytes))
    }

    pub(crate) fn visible_bytes(&self) -> u64 {
        self.visible.iter().fold(0_u64, |total, index| {
            total.saturating_add(self.models[*index].size_bytes)
        })
    }

    pub(crate) fn metadata_for(&self, model: &ModelInfo) -> Option<&LocalModelMetadata> {
        self.local_metadata.get(&model.id)
    }

    pub(crate) fn context_estimate(&self, model: &ModelInfo) -> ContextVramEstimate {
        let fallback = LocalModelMetadata::default();
        model.context_vram_estimate(
            self.metadata_for(model).unwrap_or(&fallback),
            self.context_tokens,
            VRAM_OVERHEAD_PERCENT,
        )
    }

    pub(crate) fn estimated_vram_mib(&self, model: &ModelInfo) -> Option<u64> {
        let bytes = self.context_estimate(model).allocation_input_bytes?;
        bytes.checked_add((1 << 20) - 1).map(|value| value >> 20)
    }

    pub(crate) fn gpu_allocation_estimate(
        &self,
        model: &ModelInfo,
    ) -> Option<GpuAllocationEstimate> {
        let required_bytes = self.context_estimate(model).allocation_input_bytes?;
        Some(self.gpu_detection.allocate(required_bytes))
    }

    pub(crate) fn gpu_count_estimate(&self, model: &ModelInfo) -> GpuCountEstimate {
        let Some(required_mib) = self.estimated_vram_mib(model) else {
            return GpuCountEstimate::Unknown;
        };
        self.gpu_detection.gpus_needed(required_mib)
    }

    fn refresh(&mut self, refresh_gpus: bool) {
        let selected_id = self.selected_model().map(|model| model.id.clone());
        if refresh_gpus {
            self.gpu_detection = gpu::detect_nvidia_gpus();
        }
        match cache::scan_models(&self.cache_root) {
            Ok(models) => {
                self.models = models;
                self.local_metadata = self
                    .models
                    .iter()
                    .map(|model| (model.id.clone(), model.local_metadata()))
                    .collect();
                self.scan_error = None;
                self.rebuild_visible(selected_id.as_deref());
                self.restart_hub_enrichment();
            }
            Err(error) => {
                self.scan_error = Some(format!(
                    "Could not scan {}: {error}",
                    self.cache_root.display()
                ));
                self.rebuild_visible(selected_id.as_deref());
            }
        }
    }

    fn restart_hub_enrichment(&mut self) {
        self.hub_generation = self.hub_generation.wrapping_add(1);
        self.hub_states.clear();
        let Some(worker) = &self.hub_worker else {
            if let Some(error) = &self.hub_worker_error {
                for model in &self.models {
                    self.hub_states
                        .insert(model.id.clone(), HubModelState::Error(error.clone()));
                }
            }
            return;
        };
        worker
            .latest_generation
            .store(self.hub_generation, AtomicOrdering::Release);
        for model in &self.models {
            self.hub_states
                .insert(model.id.clone(), HubModelState::Loading);
        }
        worker.replace_pending(HubJob {
            generation: self.hub_generation,
            model_ids: self.models.iter().map(|model| model.id.clone()).collect(),
        });
    }

    fn apply_hub_result(&mut self, result: HubResult) -> bool {
        if result.generation != self.hub_generation
            || !self.models.iter().any(|model| model.id == result.model_id)
        {
            return false;
        }
        let state = match result.result {
            Ok(metadata) => HubModelState::Ready(metadata),
            Err(error) => HubModelState::Error(error.to_string()),
        };
        self.hub_states.insert(result.model_id, state);
        true
    }

    fn drain_hub_results(&mut self) -> bool {
        let mut changed = false;
        let mut results = Vec::new();
        if let Some(worker) = &self.hub_worker {
            while let Ok(result) = worker.results.try_recv() {
                results.push(result);
            }
        }
        for result in results {
            changed |= self.apply_hub_result(result);
        }
        changed
    }

    pub(crate) fn hub_progress(&self) -> (usize, usize) {
        let loading = self
            .hub_states
            .values()
            .filter(|state| matches!(state, HubModelState::Loading))
            .count();
        (
            self.hub_states.len().saturating_sub(loading),
            self.hub_states.len(),
        )
    }

    fn cycle_context(&mut self) {
        let maximum = self
            .selected_model()
            .and_then(|model| self.metadata_for(model))
            .and_then(|metadata| metadata.maximum_context_length);
        let presets = cache::context_presets(maximum);
        if presets.is_empty() {
            return;
        }
        self.context_tokens = presets
            .iter()
            .copied()
            .find(|value| *value > self.context_tokens)
            .unwrap_or(presets[0]);
    }

    fn rebuild_visible(&mut self, preferred_id: Option<&str>) {
        let query = self.filter.trim().to_lowercase();
        self.visible = self
            .models
            .iter()
            .enumerate()
            .filter(|(_, model)| {
                query.is_empty()
                    || model.id.to_lowercase().contains(&query)
                    || model.organization.to_lowercase().contains(&query)
            })
            .map(|(index, _)| index)
            .collect();

        let models = &self.models;
        match self.sort {
            SortMode::Name => self
                .visible
                .sort_by(|a, b| models[*a].id.cmp(&models[*b].id)),
            SortMode::Size => self.visible.sort_by(|a, b| {
                models[*b]
                    .size_bytes
                    .cmp(&models[*a].size_bytes)
                    .then_with(|| models[*a].id.cmp(&models[*b].id))
            }),
            SortMode::Modified => self.visible.sort_by(|a, b| {
                compare_modified(models[*b].last_modified, models[*a].last_modified)
                    .then_with(|| models[*a].id.cmp(&models[*b].id))
            }),
        }
        if self.sort_reversed {
            self.visible.reverse();
        }

        self.selected = preferred_id
            .and_then(|id| {
                self.visible
                    .iter()
                    .position(|index| self.models[*index].id == id)
            })
            .unwrap_or_else(|| self.selected.min(self.visible.len().saturating_sub(1)));
    }

    fn move_selection(&mut self, amount: isize) {
        if self.visible.is_empty() {
            self.selected = 0;
            return;
        }
        self.selected = self
            .selected
            .saturating_add_signed(amount)
            .min(self.visible.len() - 1);
    }

    fn handle_key(&mut self, key: KeyEvent, page_size: usize) -> Option<AppAction> {
        if key.kind != KeyEventKind::Press {
            return None;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.should_quit = true;
            return None;
        }
        if self.pending_delete.is_some() {
            let pending = self.pending_delete.take().expect("pending delete checked");
            if matches!(key.code, KeyCode::Char('y' | 'Y'))
                && matches!(key.modifiers, KeyModifiers::NONE | KeyModifiers::SHIFT)
            {
                self.deleting = Some(pending.clone());
                return Some(AppAction::Delete(Box::new(pending)));
            }
            return None;
        }
        if self.editing_filter {
            match key.code {
                KeyCode::Esc => {
                    self.editing_filter = false;
                    self.filter_draft = self.filter.clone();
                }
                KeyCode::Enter => {
                    let selected_id = self.selected_model().map(|model| model.id.clone());
                    self.filter.clone_from(&self.filter_draft);
                    self.editing_filter = false;
                    self.rebuild_visible(selected_id.as_deref());
                }
                KeyCode::Backspace => {
                    self.filter_draft.pop();
                }
                KeyCode::Char(character)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    self.filter_draft.push(character);
                }
                _ => {}
            }
            return None;
        }

        if self.show_help {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q')
            ) {
                self.show_help = false;
            }
            return None;
        }

        match key.code {
            KeyCode::Char('q') => {
                self.action_status = None;
                self.should_quit = true;
            }
            KeyCode::Esc if !self.filter.is_empty() => {
                self.action_status = None;
                let selected_id = self.selected_model().map(|model| model.id.clone());
                self.filter.clear();
                self.filter_draft.clear();
                self.rebuild_visible(selected_id.as_deref());
            }
            KeyCode::Char('?') => {
                self.action_status = None;
                self.show_help = true;
            }
            KeyCode::Char('/') => {
                self.action_status = None;
                self.editing_filter = true;
                self.filter_draft.clone_from(&self.filter);
            }
            KeyCode::Char('r') => {
                self.action_status = None;
                return Some(AppAction::Refresh);
            }
            KeyCode::Char('c') => {
                self.action_status = None;
                self.cycle_context();
            }
            KeyCode::Char('d') => {
                let model = self.selected_model()?.clone();
                self.action_status = None;
                self.pending_delete = Some(PendingDelete {
                    id: model.id.clone(),
                    path: model.path.clone(),
                    size_bytes: model.size_bytes,
                    model,
                    visible_index: self.selected,
                });
                return Some(AppAction::DrainInput);
            }
            KeyCode::Char('s') => {
                self.action_status = None;
                let selected_id = self.selected_model().map(|model| model.id.clone());
                self.sort = self.sort.next();
                self.sort_reversed = false;
                self.rebuild_visible(selected_id.as_deref());
            }
            KeyCode::Char('S') => {
                self.action_status = None;
                let selected_id = self.selected_model().map(|model| model.id.clone());
                self.sort_reversed = !self.sort_reversed;
                self.rebuild_visible(selected_id.as_deref());
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.action_status = None;
                self.move_selection(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.action_status = None;
                self.move_selection(-1);
            }
            KeyCode::PageDown => {
                self.action_status = None;
                self.move_selection(page_size.max(1) as isize);
            }
            KeyCode::PageUp => {
                self.action_status = None;
                self.move_selection(-(page_size.max(1) as isize));
            }
            KeyCode::Home => {
                self.action_status = None;
                self.selected = 0;
            }
            KeyCode::End => {
                self.action_status = None;
                self.selected = self.visible.len().saturating_sub(1);
            }
            _ => {}
        }
        None
    }

    fn finish_delete(&mut self, pending: &PendingDelete, result: std::io::Result<()>) {
        self.deleting = None;
        match result {
            Ok(()) => {
                self.models.retain(|model| model.path != pending.path);
                self.selected = pending.visible_index;
                self.rebuild_visible(None);
                self.action_status = Some(ActionStatus::Success(format!(
                    "Deleted {} and reclaimed {} bytes",
                    pending.id, pending.size_bytes
                )));
            }
            Err(error) => {
                self.action_status = Some(ActionStatus::Failure(format!(
                    "Failed to delete {}: {} - the cache directory may be partially removed: {}",
                    pending.id,
                    error,
                    pending.path.display()
                )));
            }
        }
    }
}

fn compare_modified(left: Option<SystemTime>, right: Option<SystemTime>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn run(terminal: &mut Tui, cache_dir: Option<PathBuf>, online: bool) -> Result<()> {
    let cache_root = cache_dir
        .or_else(cache::discover_cache_root)
        .unwrap_or_else(|| PathBuf::from(".cache/huggingface/hub"));
    let mut app = App::new_with_online(cache_root, gpu::detect_nvidia_gpus(), online);
    app.refresh(false);
    let mut dirty = true;

    while !app.should_quit {
        dirty |= app.drain_hub_results();
        if dirty {
            terminal.draw(|frame| ui::draw(frame, &app))?;
            dirty = false;
        }

        if event::poll(EVENT_WAIT)? {
            match event::read()? {
                Event::Key(key) => {
                    let rows = terminal.size()?.height.saturating_sub(16) as usize;
                    match app.handle_key(key, rows) {
                        Some(AppAction::Refresh) => app.refresh(true),
                        Some(AppAction::DrainInput) => {
                            while event::poll(Duration::ZERO)? {
                                let _ = event::read()?;
                            }
                        }
                        Some(AppAction::Delete(pending)) => {
                            terminal.draw(|frame| ui::draw(frame, &app))?;
                            let result = cache::delete_model(&app.cache_root, &pending.model);
                            if result.is_err() {
                                app.refresh(false);
                            }
                            app.finish_delete(&pending, result);
                            if matches!(app.action_status, Some(ActionStatus::Success(_))) {
                                app.refresh(false);
                            }
                        }
                        None => {}
                    }
                    dirty = true;
                }
                Event::Resize(_, _) => dirty = true,
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{
        CacheCompleteness, ModelDtype, ModelIdentity, PrecisionInfo, TransformerDimensions,
        WeightFormat,
    };
    use ratatui::{Terminal, backend::TestBackend};

    fn model(id: &str, bytes: u64) -> ModelInfo {
        let (organization, name) = id.split_once('/').unwrap();
        ModelInfo {
            id: id.into(),
            organization: organization.into(),
            name: name.into(),
            path: id.into(),
            size_bytes: bytes,
            estimated_model_weight_bytes: Some(bytes),
            snapshot_count: 1,
            revision_count: 1,
            last_modified: None,
        }
    }

    fn app() -> App {
        let mut app = App::new("cache".into(), GpuDetection::NoGpu);
        app.models = vec![
            model("zeta/small", 10),
            model("acme/large", 100),
            model("acme/tiny", 1),
        ];
        app.rebuild_visible(None);
        app
    }

    fn decoder_metadata(maximum_context_length: Option<u64>) -> LocalModelMetadata {
        LocalModelMetadata {
            identity: ModelIdentity {
                architectures: vec!["LlamaForCausalLM".into()],
                model_type: Some("llama".into()),
                family: Some("Llama".into()),
            },
            precision: PrecisionInfo {
                dtype: Some(ModelDtype::Float16),
                quantization: None,
            },
            maximum_context_length,
            pipeline_task: Some("text-generation".into()),
            weight_formats: vec![WeightFormat::SafeTensors],
            completeness: CacheCompleteness::Complete,
            transformer: TransformerDimensions {
                hidden_size: Some(4096),
                num_hidden_layers: Some(32),
                num_attention_heads: Some(32),
                num_key_value_heads: Some(8),
                head_dim: Some(128),
                dtype_bytes: Some(2),
            },
            ..LocalModelMetadata::default()
        }
    }

    fn press(character: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(character), KeyModifiers::NONE)
    }

    #[test]
    fn defaults_to_largest_models_first() {
        let app = app();
        assert_eq!(app.sort, SortMode::Size);
        assert_eq!(app.models[app.visible[0]].id, "acme/large");
        assert_eq!(app.models[app.visible[2]].id, "acme/tiny");
    }

    #[test]
    fn filtering_is_case_insensitive_and_preserves_selection() {
        let mut app = app();
        app.selected = 1;
        assert_eq!(app.selected_model().unwrap().id, "zeta/small");
        app.filter = "ACME".into();
        app.rebuild_visible(Some("acme/large"));
        assert_eq!(app.visible.len(), 2);
        assert_eq!(app.selected_model().unwrap().id, "acme/large");
    }

    #[test]
    fn sorting_cycles_and_preserves_selected_model() {
        let mut app = app();
        let selected = app.selected_model().unwrap().id.clone();
        app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE), 5);
        assert_eq!(app.sort, SortMode::Modified);
        assert_eq!(app.selected_model().unwrap().id, selected);
    }

    #[test]
    fn uppercase_s_reverses_the_current_sort() {
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('S'), KeyModifiers::SHIFT), 5);
        assert!(app.sort_reversed);
        assert_eq!(app.models[app.visible[0]].id, "acme/tiny");

        app.handle_key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE), 5);
        assert_eq!(app.sort, SortMode::Modified);
        assert!(!app.sort_reversed);
    }

    #[test]
    fn filter_edit_can_be_cancelled_or_applied() {
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE), 5);
        app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE), 5);
        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), 5);
        assert!(app.filter.is_empty());
        assert!(!app.editing_filter);

        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE), 5);
        app.handle_key(KeyEvent::new(KeyCode::Char('z'), KeyModifiers::NONE), 5);
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE), 5);
        assert_eq!(app.filter, "z");
        assert_eq!(app.visible.len(), 1);

        app.handle_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE), 5);
        assert!(app.filter.is_empty());
        assert_eq!(app.visible.len(), 3);
    }

    #[test]
    fn control_c_quits_from_filter_input() {
        let mut app = app();
        app.handle_key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE), 5);
        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL), 5);
        assert!(app.should_quit);
    }

    #[test]
    fn gpu_count_preserves_no_gpu_unknown_and_insufficient_states() {
        let mut app = app();
        let model = model("acme/large", 10 * 1024 * 1024);
        app.local_metadata
            .insert(model.id.clone(), decoder_metadata(None));

        assert!(matches!(
            app.gpu_count_estimate(&model),
            GpuCountEstimate::InsufficientCapacity {
                available_mib: 0,
                ..
            }
        ));

        app.gpu_detection = GpuDetection::ToolUnavailable {
            kind: std::io::ErrorKind::NotFound,
            message: "missing".into(),
        };
        assert_eq!(app.gpu_count_estimate(&model), GpuCountEstimate::Unknown);

        app.gpu_detection = GpuDetection::Detected(
            gpu::parse_nvidia_smi_output("small, 8, 8")
                .unwrap()
                .unwrap(),
        );
        assert!(matches!(
            app.gpu_count_estimate(&model),
            GpuCountEstimate::InsufficientCapacity {
                available_mib: 8,
                ..
            }
        ));

        app.gpu_detection = GpuDetection::Detected(
            gpu::parse_nvidia_smi_output("large, 512, 512")
                .unwrap()
                .unwrap(),
        );
        assert!(matches!(
            app.gpu_count_estimate(&model),
            GpuCountEstimate::Gpus(count) if count.get() == 1
        ));
    }

    #[test]
    fn offline_is_default_and_never_starts_a_hub_worker() {
        let app = App::new("cache".into(), GpuDetection::NoGpu);
        assert!(!app.online);
        assert!(app.hub_worker.is_none());
        assert!(app.hub_worker_error.is_none());
        assert!(app.hub_states.is_empty());
    }

    #[test]
    fn stale_hub_results_are_rejected() {
        let mut app = app();
        app.hub_generation = 7;
        app.hub_states
            .insert("acme/large".into(), HubModelState::Loading);
        let result = HubResult {
            generation: 6,
            model_id: "acme/large".into(),
            result: Ok(ModelMetadata {
                repo_id: "acme/large".into(),
                latest_revision: Some("abcdef0123456789".into()),
                last_modified: None,
                pipeline_tag: None,
                library: None,
                tags: vec![],
                license: None,
                gated: hub::GatedStatus::No,
                private: false,
                disabled: false,
                deprecated: false,
            }),
        };
        assert!(!app.apply_hub_result(result));
        assert_eq!(
            app.hub_states.get("acme/large"),
            Some(&HubModelState::Loading)
        );
    }

    #[test]
    fn context_key_changes_vram_and_tab_keys_have_no_effect() {
        let mut app = app();
        let id = app.selected_model().unwrap().id.clone();
        app.local_metadata
            .insert(id, decoder_metadata(Some(32_768)));
        let before = app
            .context_estimate(app.selected_model().unwrap())
            .allocation_input_bytes
            .unwrap();

        let selected = app.selected;
        let context = app.context_tokens;
        assert!(
            app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE), 5)
                .is_none()
        );
        assert!(
            app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT), 5)
                .is_none()
        );
        assert_eq!(app.selected, selected);
        assert_eq!(app.context_tokens, context);

        app.handle_key(press('c'), 5);
        assert_eq!(app.context_tokens, 4_096);
        let after = app
            .context_estimate(app.selected_model().unwrap())
            .allocation_input_bytes
            .unwrap();
        assert!(after > before);
    }

    #[test]
    fn model_max_context_is_deduplicated_and_caps_estimates() {
        let mut app = app();
        let id = app.selected_model().unwrap().id.clone();
        app.local_metadata.insert(id, decoder_metadata(Some(3_000)));
        app.handle_key(press('c'), 5);
        assert_eq!(app.context_tokens, 3_000);
        app.context_tokens = 8_192;
        let estimate = app.context_estimate(app.selected_model().unwrap());
        assert_eq!(estimate.effective_tokens, 3_000);
        assert!(estimate.capped_to_model_max);
    }

    #[test]
    fn model_fixture_initializes_runtime_weight_estimate() {
        let model = model("acme/model", 42);
        assert_eq!(model.estimated_model_weight_bytes, Some(42));
    }

    #[test]
    fn delete_confirmation_captures_the_exact_selected_model() {
        let mut app = app();
        app.selected = 1;
        let selected = app.selected_model().unwrap().clone();

        assert!(matches!(
            app.handle_key(press('d'), 5),
            Some(AppAction::DrainInput)
        ));
        let pending = app.pending_delete.as_ref().unwrap();
        assert_eq!(pending.id, selected.id);
        assert_eq!(pending.path, selected.path);
        assert_eq!(pending.size_bytes, selected.size_bytes);
    }

    #[test]
    fn lowercase_and_uppercase_y_confirm_and_transition_to_deleting() {
        for key in [
            press('y'),
            KeyEvent::new(KeyCode::Char('Y'), KeyModifiers::SHIFT),
            KeyEvent::new(KeyCode::Char('Y'), KeyModifiers::NONE),
        ] {
            let mut app = app();
            app.handle_key(press('d'), 5);

            let action = app.handle_key(key, 5);

            assert!(matches!(action, Some(AppAction::Delete(_))));
            assert!(app.pending_delete.is_none());
            assert_eq!(app.deleting.as_ref().unwrap().id, "acme/large");
        }
    }

    #[test]
    fn every_key_other_than_unmodified_y_or_shift_y_cancels_without_quitting() {
        for key in [
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            press('n'),
            press('q'),
            press('x'),
            KeyEvent::new(KeyCode::Char('y'), KeyModifiers::CONTROL),
            KeyEvent::new(KeyCode::Char('Y'), KeyModifiers::ALT),
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
        ] {
            let mut app = app();
            app.handle_key(press('d'), 5);

            assert!(app.handle_key(key, 5).is_none());
            assert!(app.pending_delete.is_none());
            assert!(!app.should_quit);
            assert_eq!(app.selected, 0);
        }
    }

    #[test]
    fn control_c_remains_global_while_confirmation_is_open() {
        let mut app = app();
        app.handle_key(press('d'), 5);

        app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL), 5);

        assert!(app.should_quit);
        assert!(app.pending_delete.is_some());
    }

    #[test]
    fn delete_is_filter_input_and_is_ignored_in_help() {
        let mut app = app();
        app.handle_key(press('/'), 5);
        app.handle_key(press('d'), 5);
        assert_eq!(app.filter_draft, "d");
        assert!(app.pending_delete.is_none());

        app.editing_filter = false;
        app.show_help = true;
        app.handle_key(press('d'), 5);
        assert!(app.show_help);
        assert!(app.pending_delete.is_none());
    }

    #[test]
    fn pending_confirmation_blocks_other_actions() {
        for key in [press('?'), press('/'), press('r'), press('s'), press('j')] {
            let mut app = app();
            let selected = app.selected;
            let sort = app.sort;
            app.handle_key(press('d'), 5);

            assert!(app.handle_key(key, 5).is_none());
            assert!(!app.show_help);
            assert!(!app.editing_filter);
            assert_eq!(app.sort, sort);
            assert_eq!(app.selected, selected);
        }
    }

    #[test]
    fn delete_is_a_noop_without_a_visible_model() {
        let mut empty_app = app();
        empty_app.models.clear();
        empty_app.rebuild_visible(None);
        assert!(empty_app.handle_key(press('d'), 5).is_none());
        assert!(empty_app.pending_delete.is_none());

        let mut filtered_app = app();
        filtered_app.filter = "no-match".into();
        filtered_app.rebuild_visible(None);
        assert!(filtered_app.handle_key(press('d'), 5).is_none());
        assert!(filtered_app.pending_delete.is_none());
    }

    #[test]
    fn successful_delete_preserves_filter_and_selects_an_adjacent_model() {
        let mut filtered_app = app();
        filtered_app.filter = "acme".into();
        filtered_app.rebuild_visible(None);
        filtered_app.selected = 0;
        filtered_app.handle_key(press('d'), 5);
        let pending = filtered_app.pending_delete.take().unwrap();

        filtered_app.finish_delete(&pending, Ok(()));

        assert_eq!(filtered_app.filter, "acme");
        assert_eq!(filtered_app.visible.len(), 1);
        assert_eq!(filtered_app.selected_model().unwrap().id, "acme/tiny");
        assert!(matches!(
            filtered_app.action_status,
            Some(ActionStatus::Success(_))
        ));

        let mut last_app = app();
        last_app.selected = last_app.visible.len() - 1;
        last_app.handle_key(press('d'), 5);
        let pending = last_app.pending_delete.take().unwrap();
        last_app.finish_delete(&pending, Ok(()));
        assert_eq!(last_app.selected, last_app.visible.len() - 1);
    }

    #[test]
    fn deletion_failure_status_survives_refresh_until_normal_input() {
        let mut app = app();
        app.handle_key(press('d'), 5);
        let pending = app.pending_delete.take().unwrap();
        app.finish_delete(
            &pending,
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "denied",
            )),
        );
        let expected = format!(
            "Failed to delete {}: denied - the cache directory may be partially removed: {}",
            pending.id,
            pending.path.display()
        );
        assert_eq!(app.action_status.as_ref().unwrap().message(), expected);

        app.refresh(false);
        assert_eq!(app.action_status.as_ref().unwrap().message(), expected);

        app.handle_key(press('x'), 5);
        assert!(app.action_status.is_some());
        app.handle_key(press('j'), 5);
        assert!(app.action_status.is_none());
    }

    #[test]
    fn tiny_rendering_handles_help_delete_and_deleting_overlays() {
        for (width, height) in [(1, 1), (10, 3), (30, 10)] {
            let mut app = app();
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();

            app.show_help = true;
            terminal.draw(|frame| ui::draw(frame, &app)).unwrap();

            app.show_help = false;
            app.handle_key(press('d'), 1);
            terminal.draw(|frame| ui::draw(frame, &app)).unwrap();

            app.handle_key(press('y'), 1);
            terminal.draw(|frame| ui::draw(frame, &app)).unwrap();
        }
    }

    #[test]
    fn one_cell_delete_confirmation_still_shows_emphasized_y() {
        let mut app = app();
        app.handle_key(press('d'), 1);
        let mut terminal = Terminal::new(TestBackend::new(1, 1)).unwrap();

        terminal.draw(|frame| ui::draw(frame, &app)).unwrap();

        let cell = &terminal.backend().buffer().content()[0];
        assert_eq!(cell.symbol(), "Y");
        assert_eq!(cell.fg, ratatui::style::Color::Yellow);
        assert!(cell.modifier.contains(ratatui::style::Modifier::BOLD));
        assert!(cell.modifier.contains(ratatui::style::Modifier::UNDERLINED));
    }

    #[test]
    fn delete_confirmation_copy_is_visible_in_normal_compact_narrow_and_short_views() {
        for (width, height) in [(100, 24), (45, 14), (34, 10), (80, 2)] {
            let mut app = app();
            app.handle_key(press('d'), 1);
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();

            terminal.draw(|frame| ui::draw(frame, &app)).unwrap();

            let text: String = terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|cell| cell.symbol())
                .collect();
            assert!(
                text.contains("Press Y to permanently delete"),
                "missing confirmation instruction at {width}x{height}: {text:?}"
            );
            assert!(
                text.contains("Any other key cancels"),
                "missing cancellation instruction at {width}x{height}: {text:?}"
            );
            assert!(
                terminal.backend().buffer().content().iter().any(|cell| {
                    cell.symbol() == "Y"
                        && cell.fg == ratatui::style::Color::Yellow
                        && cell.bg == ratatui::style::Color::DarkGray
                        && cell.modifier.contains(ratatui::style::Modifier::BOLD)
                        && cell.modifier.contains(ratatui::style::Modifier::UNDERLINED)
                }),
                "Y is not visually emphasized at {width}x{height}"
            );
        }
    }
}
