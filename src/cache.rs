//! Discovery and inspection of the on-disk Hugging Face Hub cache.
//!
//! This module deliberately uses only the standard library. Cache entries are
//! treated as untrusted filesystem data: malformed names and unreadable
//! children are skipped rather than making the whole scan fail.

use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{OsStr, OsString},
    fs,
    io::{self, Read, Seek, SeekFrom},
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

use serde_json::Value;

use crate::gpu::{ComputeCapability, GpuDetection};

const MODEL_PREFIX: &str = "models--";
const MAX_JSON_BYTES: u64 = 2 * 1024 * 1024;
const MAX_MODEL_CARD_BYTES: u64 = 512 * 1024;
const MAX_WEIGHT_HEADER_BYTES: u64 = 16 * 1024 * 1024;
/// Upper bound on a `refs` file, which holds a single commit hash.
const MAX_REFERENCE_BYTES: u64 = 4096;
#[cfg(test)]
const BYTES_PER_MIB: u64 = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelMetadata {
    pub identity: ModelIdentity,
    pub parameters: Option<ParameterCount>,
    pub precision: PrecisionInfo,
    pub maximum_context_length: Option<u64>,
    pub pipeline_task: Option<String>,
    pub license: Option<String>,
    pub weight_formats: Vec<WeightFormat>,
    pub selected_revision: Option<SelectedRevision>,
    pub completeness: CacheCompleteness,
    pub warnings: Vec<CacheWarning>,
    pub transformer: TransformerDimensions,
    pub requires_remote_code: bool,
}

impl Default for LocalModelMetadata {
    fn default() -> Self {
        Self {
            identity: ModelIdentity::default(),
            parameters: None,
            precision: PrecisionInfo::default(),
            maximum_context_length: None,
            pipeline_task: None,
            license: None,
            weight_formats: Vec::new(),
            selected_revision: None,
            completeness: CacheCompleteness::Unknown,
            warnings: Vec::new(),
            transformer: TransformerDimensions::default(),
            requires_remote_code: false,
        }
    }
}

/// Config dimensions needed for a batch-size-one transformer KV-cache estimate.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TransformerDimensions {
    pub hidden_size: Option<u64>,
    pub num_hidden_layers: Option<u64>,
    pub num_attention_heads: Option<u64>,
    pub num_key_value_heads: Option<u64>,
    pub head_dim: Option<u64>,
    pub dtype_bytes: Option<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EstimateConfidence {
    Exact,
    Estimated,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextVramEstimate {
    pub requested_tokens: u64,
    pub effective_tokens: u64,
    pub capped_to_model_max: bool,
    pub base_allocation_bytes: Option<u64>,
    pub kv_cache_bytes: Option<u64>,
    pub allocation_input_bytes: Option<u64>,
    pub confidence: EstimateConfidence,
    pub reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompatibilityStatus {
    Compatible,
    LikelyCompatible,
    Warning,
    Unsupported,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompatibilityIssue {
    pub status: CompatibilityStatus,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCompatibility {
    pub status: CompatibilityStatus,
    pub issues: Vec<CompatibilityIssue>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelIdentity {
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub family: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterCount {
    pub value: u64,
    pub confidence: ParameterCountConfidence,
    pub source: MetadataSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterCountConfidence {
    Exact,
    Reported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MetadataSource {
    SafeTensorsHeader,
    SafeTensorsMetadata,
    ConfigField(String),
    ModelCard,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PrecisionInfo {
    pub dtype: Option<ModelDtype>,
    pub quantization: Option<QuantizationInfo>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ModelDtype {
    Float64,
    Float32,
    Float16,
    BFloat16,
    Float8,
    Int8,
    Int4,
    Other(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizationInfo {
    pub method: String,
    pub bits: Option<u8>,
    pub variant: Option<String>,
    pub source: QuantizationSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantizationSource {
    Config,
    FileName,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelectedRevision {
    pub commit: String,
    pub references: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheCompleteness {
    Complete,
    Partial,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CacheWarning {
    MissingReferencedFile(PathBuf),
    IncompleteArtifact(PathBuf),
    LockArtifact(PathBuf),
    MalformedMetadata(PathBuf),
    OversizedMetadata(PathBuf),
    IncompleteShardSet {
        group: String,
        expected: usize,
        found: usize,
    },
    MultipleSnapshots(usize),
    DuplicateRevisionReferences {
        commit: String,
        references: Vec<String>,
    },
}

/// Information about one locally cached Hugging Face model repository.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    /// Hub repository id in `organization/name` form.
    pub id: String,
    pub organization: String,
    pub name: String,
    /// Directory containing this model's `blobs`, `refs`, and `snapshots`.
    pub path: PathBuf,
    /// Physical bytes stored below [`Self::path`].
    ///
    /// Symbolic links are not followed, so snapshot links to files in `blobs`
    /// do not count the same content a second time.
    pub size_bytes: u64,
    /// Estimated bytes occupied by one coherent set of model weights.
    ///
    /// This is derived from recognized weight artifacts in one selected
    /// snapshot, not from total cache size. It is a useful lower-bound proxy
    /// for runtime VRAM, but excludes activations, KV cache, allocator and
    /// framework overhead. `None` means no usable local weight artifacts were
    /// found.
    pub estimated_model_weight_bytes: Option<u64>,
    /// Number of snapshot directories in the model cache.
    pub snapshot_count: usize,
    /// Number of readable reference files below `refs`.
    pub revision_count: usize,
    /// Most recent filesystem modification time found in the model cache.
    pub last_modified: Option<SystemTime>,
}

impl ModelInfo {
    /// Inspect the selected local snapshot without performing network access.
    ///
    /// Metadata is evaluated on demand so callers can rescan a cache entry
    /// after a download progresses without reconstructing `ModelInfo`.
    pub fn local_metadata(&self) -> LocalModelMetadata {
        inspect_local_metadata(&self.path, self.snapshot_count)
    }

    /// Estimate allocation bytes for weights, the existing runtime allowance,
    /// and a context-dependent KV cache.
    pub fn context_vram_estimate(
        &self,
        metadata: &LocalModelMetadata,
        requested_tokens: u64,
        runtime_allowance_percent: u32,
    ) -> ContextVramEstimate {
        metadata.context_vram_estimate(
            self.estimated_model_weight_bytes,
            requested_tokens,
            runtime_allowance_percent,
        )
    }
}

impl LocalModelMetadata {
    pub fn context_vram_estimate(
        &self,
        weight_bytes: Option<u64>,
        requested_tokens: u64,
        runtime_allowance_percent: u32,
    ) -> ContextVramEstimate {
        let effective_tokens = self
            .maximum_context_length
            .map_or(requested_tokens, |maximum| requested_tokens.min(maximum));
        let capped_to_model_max = effective_tokens != requested_tokens;
        let base_allocation_bytes =
            weight_bytes.and_then(|bytes| add_overhead_bytes(bytes, runtime_allowance_percent));

        if weight_bytes.is_some() && base_allocation_bytes.is_none() {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                None,
                "weight/runtime allowance exceeds supported size",
            );
        }
        let Some(base_allocation_bytes) = base_allocation_bytes else {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                None,
                "local weight size is unavailable",
            );
        };
        if !self.supports_transformer_kv_cache() {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                Some(base_allocation_bytes),
                "model task or architecture does not support a defensible decoder KV-cache formula",
            );
        }

        let dimensions = &self.transformer;
        let (Some(layers), Some(attention_heads), Some(dtype_bytes)) = (
            dimensions.num_hidden_layers,
            dimensions.num_attention_heads,
            dimensions.dtype_bytes,
        ) else {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                Some(base_allocation_bytes),
                "required transformer dimensions or KV dtype are unavailable",
            );
        };
        let (kv_heads, used_mha_fallback) = match dimensions.num_key_value_heads {
            Some(value) => (value, false),
            None => (attention_heads, true),
        };
        let (head_dim, derived_head_dim) = match dimensions.head_dim {
            Some(value) => (value, false),
            None => {
                let Some(hidden_size) = dimensions.hidden_size else {
                    return unavailable_context_estimate(
                        requested_tokens,
                        effective_tokens,
                        capped_to_model_max,
                        Some(base_allocation_bytes),
                        "head_dim and hidden_size are unavailable",
                    );
                };
                if hidden_size % attention_heads != 0 {
                    return unavailable_context_estimate(
                        requested_tokens,
                        effective_tokens,
                        capped_to_model_max,
                        Some(base_allocation_bytes),
                        "hidden_size is not divisible by num_attention_heads",
                    );
                }
                (hidden_size / attention_heads, true)
            }
        };
        if layers == 0 || attention_heads == 0 || kv_heads == 0 || head_dim == 0 {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                Some(base_allocation_bytes),
                "transformer dimensions must be non-zero",
            );
        }

        let kv_cache_bytes = [
            2,
            effective_tokens,
            layers,
            kv_heads,
            head_dim,
            u64::from(dtype_bytes),
        ]
        .into_iter()
        .try_fold(1_u64, u64::checked_mul);
        let Some(kv_cache_bytes) = kv_cache_bytes else {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                Some(base_allocation_bytes),
                "KV-cache size exceeds supported size",
            );
        };
        let Some(allocation_input_bytes) = base_allocation_bytes.checked_add(kv_cache_bytes) else {
            return unavailable_context_estimate(
                requested_tokens,
                effective_tokens,
                capped_to_model_max,
                Some(base_allocation_bytes),
                "combined allocation size exceeds supported size",
            );
        };

        ContextVramEstimate {
            requested_tokens,
            effective_tokens,
            capped_to_model_max,
            base_allocation_bytes: Some(base_allocation_bytes),
            kv_cache_bytes: Some(kv_cache_bytes),
            allocation_input_bytes: Some(allocation_input_bytes),
            confidence: if used_mha_fallback || derived_head_dim {
                EstimateConfidence::Estimated
            } else {
                EstimateConfidence::Exact
            },
            reason: capped_to_model_max
                .then(|| "requested context was capped to the model maximum".to_owned()),
        }
    }

    pub fn runtime_compatibility(&self, gpu: &GpuDetection) -> RuntimeCompatibility {
        let mut issues = Vec::new();
        if self.completeness == CacheCompleteness::Partial {
            issues.push(issue(
                CompatibilityStatus::Unsupported,
                "local weights are incomplete",
            ));
        } else if self.completeness == CacheCompleteness::Unknown {
            issues.push(issue(
                CompatibilityStatus::Unknown,
                "weight completeness could not be established",
            ));
        }
        if self.weight_formats.is_empty() {
            issues.push(issue(
                CompatibilityStatus::Unknown,
                "no recognized local weight format",
            ));
        }
        if self.weight_formats.contains(&WeightFormat::Gguf) {
            issues.push(issue(
                CompatibilityStatus::LikelyCompatible,
                "GGUF targets a llama.cpp-style runtime, not generic CUDA weights",
            ));
        }
        if let Some(quantization) = &self.precision.quantization {
            let method = quantization.method.to_ascii_lowercase();
            if method.contains("gptq") || method.contains("awq") {
                issues.push(issue(
                    CompatibilityStatus::Warning,
                    &format!("{} requires explicit backend support", quantization.method),
                ));
            } else if !method.contains("gguf") {
                issues.push(issue(
                    CompatibilityStatus::Warning,
                    &format!(
                        "{} quantization requires runtime support",
                        quantization.method
                    ),
                ));
            }
        }
        if self.requires_remote_code {
            issues.push(issue(
                CompatibilityStatus::Warning,
                "configuration references remote custom code",
            ));
        }
        if self.identity.architectures.is_empty() && self.identity.model_type.is_none() {
            issues.push(issue(
                CompatibilityStatus::Warning,
                "model architecture is unknown",
            ));
        }
        if !self.supports_transformer_kv_cache() {
            issues.push(issue(
                CompatibilityStatus::Warning,
                "model task or architecture requires task-specific runtime support",
            ));
        }
        if self.precision.dtype == Some(ModelDtype::BFloat16)
            && !self.weight_formats.contains(&WeightFormat::Gguf)
        {
            assess_bfloat16(gpu, &mut issues);
        } else if matches!(gpu, GpuDetection::NoGpu)
            && !self.weight_formats.contains(&WeightFormat::Gguf)
        {
            issues.push(issue(
                CompatibilityStatus::Warning,
                "no NVIDIA GPU was detected for accelerator execution",
            ));
        } else if matches!(
            gpu,
            GpuDetection::ToolUnavailable { .. }
                | GpuDetection::CommandFailed { .. }
                | GpuDetection::MalformedOutput(_)
        ) {
            issues.push(issue(
                CompatibilityStatus::Unknown,
                "GPU capabilities could not be detected",
            ));
        }

        let status = issues
            .iter()
            .map(|issue| issue.status)
            .max_by_key(|status| compatibility_severity(*status))
            .unwrap_or(CompatibilityStatus::Compatible);
        RuntimeCompatibility { status, issues }
    }

    fn supports_transformer_kv_cache(&self) -> bool {
        let task = self
            .pipeline_task
            .as_deref()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if task.contains("image")
            || task.contains("vision")
            || task.contains("audio")
            || task.contains("speech")
            || [
                "feature-extraction",
                "fill-mask",
                "text-classification",
                "token-classification",
                "image-classification",
                "audio-classification",
                "automatic-speech-recognition",
            ]
            .iter()
            .any(|candidate| task == *candidate)
        {
            return false;
        }
        task == "text-generation"
            || self.identity.architectures.iter().any(|architecture| {
                let architecture = architecture.to_ascii_lowercase();
                architecture.contains("causallm")
            })
    }
}

pub fn context_presets(model_maximum: Option<u64>) -> Vec<u64> {
    let mut presets = vec![2_048, 4_096, 8_192, 16_384, 32_768];
    if let Some(maximum) = model_maximum {
        presets.retain(|value| *value <= maximum);
        if maximum > 0 {
            presets.push(maximum);
        }
    }
    presets.sort_unstable();
    presets.dedup();
    presets
}

fn unavailable_context_estimate(
    requested_tokens: u64,
    effective_tokens: u64,
    capped_to_model_max: bool,
    base_allocation_bytes: Option<u64>,
    reason: &str,
) -> ContextVramEstimate {
    ContextVramEstimate {
        requested_tokens,
        effective_tokens,
        capped_to_model_max,
        base_allocation_bytes,
        kv_cache_bytes: None,
        allocation_input_bytes: None,
        confidence: EstimateConfidence::Unavailable,
        reason: Some(reason.to_owned()),
    }
}

fn add_overhead_bytes(bytes: u64, percent: u32) -> Option<u64> {
    let multiplier = 100_u128.checked_add(u128::from(percent))?;
    let numerator = u128::from(bytes).checked_mul(multiplier)?;
    let adjusted = numerator.checked_add(99)? / 100;
    u64::try_from(adjusted).ok()
}

fn issue(status: CompatibilityStatus, reason: &str) -> CompatibilityIssue {
    CompatibilityIssue {
        status,
        reason: reason.to_owned(),
    }
}

fn compatibility_severity(status: CompatibilityStatus) -> u8 {
    match status {
        CompatibilityStatus::Compatible => 0,
        CompatibilityStatus::LikelyCompatible => 1,
        CompatibilityStatus::Unknown => 2,
        CompatibilityStatus::Warning => 3,
        CompatibilityStatus::Unsupported => 4,
    }
}

fn assess_bfloat16(gpu: &GpuDetection, issues: &mut Vec<CompatibilityIssue>) {
    match gpu {
        GpuDetection::Detected(inventory) => {
            let capabilities: Vec<ComputeCapability> = inventory
                .gpus()
                .iter()
                .filter_map(|gpu| gpu.compute_capability)
                .collect();
            let capable = capabilities
                .iter()
                .filter(|capability| **capability >= ComputeCapability::new(8, 0))
                .count();
            if capable == inventory.len() {
                return;
            }
            if capabilities.len() == inventory.len() && capable == 0 {
                issues.push(issue(
                    CompatibilityStatus::Unsupported,
                    "BF16 requires NVIDIA compute capability 8.0 or newer",
                ));
            } else if capable > 0 {
                issues.push(issue(
                    CompatibilityStatus::Warning,
                    "BF16 support is mixed across detected GPUs; use only compute capability 8.0 or newer devices",
                ));
            } else {
                issues.push(issue(
                    CompatibilityStatus::Warning,
                    "BF16 support is unknown because compute capability was not reported",
                ));
            }
        }
        GpuDetection::NoGpu => issues.push(issue(
            CompatibilityStatus::Warning,
            "BF16 GPU execution requires a suitable GPU",
        )),
        _ => issues.push(issue(
            CompatibilityStatus::Unknown,
            "BF16 support is unknown because GPU capabilities could not be detected",
        )),
    }
}

/// Return the Hugging Face Hub cache root selected from the environment.
///
/// Precedence follows `huggingface_hub`: `HF_HUB_CACHE`, then
/// `HF_HOME/hub`, then `XDG_CACHE_HOME/huggingface/hub`, and finally
/// `HOME/.cache/huggingface/hub`. Empty variables are ignored.
pub fn discover_cache_root() -> Option<PathBuf> {
    cache_root_from(|name| env::var_os(name))
}

/// Scan a Hugging Face Hub cache root for model repositories.
///
/// A missing cache is equivalent to an empty cache. Failure to read the root
/// itself is returned, while malformed or unreadable entries below it are
/// ignored so one damaged download cannot hide healthy models.
pub fn scan_models(cache_root: impl AsRef<Path>) -> io::Result<Vec<ModelInfo>> {
    let entries = match fs::read_dir(cache_root.as_ref()) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };

    let mut models = Vec::new();
    for entry in entries.flatten() {
        let Some((organization, name)) = parse_model_dir_name(&entry.file_name()) else {
            continue;
        };
        if !entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            continue;
        }

        let path = entry.path();
        let stats = collect_tree_stats(&path);
        let snapshot_count = count_immediate_directories(&path.join("snapshots"));
        let revision_count = count_regular_files(&path.join("refs"));
        let estimated_model_weight_bytes = estimate_model_weight_bytes(&path);
        models.push(ModelInfo {
            id: format!("{organization}/{name}"),
            organization,
            name,
            path,
            size_bytes: stats.size_bytes,
            estimated_model_weight_bytes,
            snapshot_count,
            revision_count,
            last_modified: stats.last_modified,
        });
    }

    models.sort_unstable_by(|left, right| left.id.cmp(&right.id));
    Ok(models)
}

/// Delete one model repository from the active Hugging Face Hub cache.
///
/// The model path and identity are revalidated before deletion. In particular,
/// only a real directory that is an immediate child of `cache_root` and whose
/// `models--organization--name` directory matches `model` can be removed.
pub fn delete_model(cache_root: impl AsRef<Path>, model: &ModelInfo) -> io::Result<()> {
    let cache_root = cache_root.as_ref();
    let canonical_root = fs::canonicalize(cache_root).map_err(|error| {
        contextual_io_error(
            error,
            format!("cannot access active cache root {}", cache_root.display()),
        )
    })?;
    if !fs::metadata(&canonical_root)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "active cache root is not a directory: {}",
                cache_root.display()
            ),
        ));
    }

    let expected_name = expected_model_dir_name(model)?;
    if model
        .path
        .components()
        .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model path contains traversal components: {}",
                model.path.display()
            ),
        ));
    }
    if model.path.file_name() != Some(expected_name.as_os_str()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model path {} does not match model {} (expected directory {})",
                model.path.display(),
                model.id,
                expected_name.to_string_lossy()
            ),
        ));
    }

    let parent = model.path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model path has no cache-root parent: {}",
                model.path.display()
            ),
        )
    })?;
    let canonical_parent = fs::canonicalize(parent).map_err(|error| {
        contextual_io_error(
            error,
            format!(
                "cannot validate parent of model path {}",
                model.path.display()
            ),
        )
    })?;
    if canonical_parent != canonical_root {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing to delete model outside active cache root {}: {}",
                cache_root.display(),
                model.path.display()
            ),
        ));
    }

    let metadata = fs::symlink_metadata(&model.path).map_err(|error| {
        contextual_io_error(
            error,
            format!("cannot access model cache entry {}", model.path.display()),
        )
    })?;
    if metadata.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "refusing to delete symbolic-link model cache entry: {}",
                model.path.display()
            ),
        ));
    }
    if !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model cache entry is not a directory: {}",
                model.path.display()
            ),
        ));
    }

    let canonical_target = fs::canonicalize(&model.path).map_err(|error| {
        contextual_io_error(
            error,
            format!("cannot validate model cache entry {}", model.path.display()),
        )
    })?;
    let expected_target = canonical_root.join(&expected_name);
    if canonical_target != expected_target {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "refusing to delete model cache entry resolving outside its expected path {}: {}",
                expected_target.display(),
                model.path.display()
            ),
        ));
    }

    // Recheck the entry immediately before the only recursive operation.
    if fs::symlink_metadata(&model.path)?.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model cache entry became a symbolic link before deletion: {}",
                model.path.display()
            ),
        ));
    }
    fs::remove_dir_all(&canonical_target).map_err(|error| {
        contextual_io_error(
            error,
            format!(
                "failed to delete model cache entry {}",
                model.path.display()
            ),
        )
    })
}

fn expected_model_dir_name(model: &ModelInfo) -> io::Result<OsString> {
    let expected_id = format!("{}/{}", model.organization, model.name);
    if model.id != expected_id {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model identity is inconsistent: id {:?} does not match organization/name {:?}",
                model.id, expected_id
            ),
        ));
    }

    let directory_name = OsString::from(format!(
        "{MODEL_PREFIX}{}--{}",
        model.organization, model.name
    ));
    if parse_model_dir_name(&directory_name)
        != Some((model.organization.clone(), model.name.clone()))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "model identity cannot form a valid cache directory: {}",
                model.id
            ),
        ));
    }
    Ok(directory_name)
}

fn contextual_io_error(error: io::Error, context: String) -> io::Error {
    io::Error::new(error.kind(), format!("{context}: {error}"))
}

fn cache_root_from(mut get: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(path) = nonempty_path(get("HF_HUB_CACHE")) {
        return Some(path);
    }
    if let Some(path) = nonempty_path(get("HF_HOME")) {
        return Some(path.join("hub"));
    }
    if let Some(path) = nonempty_path(get("XDG_CACHE_HOME")) {
        return Some(path.join("huggingface").join("hub"));
    }
    nonempty_path(get("HOME")).map(|home| home.join(".cache").join("huggingface").join("hub"))
}

fn nonempty_path(value: Option<OsString>) -> Option<PathBuf> {
    value.filter(|value| !value.is_empty()).map(PathBuf::from)
}

fn parse_model_dir_name(value: &OsStr) -> Option<(String, String)> {
    let value = value.to_str()?;
    let remainder = value.strip_prefix(MODEL_PREFIX)?;
    let (organization, name) = remainder.split_once("--")?;
    if organization.is_empty() || name.is_empty() {
        return None;
    }
    Some((organization.to_owned(), name.to_owned()))
}

#[derive(Default)]
struct TreeStats {
    size_bytes: u64,
    last_modified: Option<SystemTime>,
}

fn collect_tree_stats(root: &Path) -> TreeStats {
    let mut stats = TreeStats::default();
    let mut pending = vec![root.to_path_buf()];

    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        update_modified(&mut stats.last_modified, metadata.modified().ok());

        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            continue;
        }
        if file_type.is_file() {
            stats.size_bytes = stats.size_bytes.saturating_add(metadata.len());
            continue;
        }
        if file_type.is_dir() {
            let Ok(entries) = fs::read_dir(path) else {
                continue;
            };
            pending.extend(entries.flatten().map(|entry| entry.path()));
        }
    }
    stats
}

fn update_modified(current: &mut Option<SystemTime>, candidate: Option<SystemTime>) {
    if let Some(candidate) = candidate
        && current.is_none_or(|value| candidate > value)
    {
        *current = Some(candidate);
    }
}

fn count_immediate_directories(path: &Path) -> usize {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .count()
        })
        .unwrap_or(0)
}

fn count_regular_files(root: &Path) -> usize {
    let mut count = 0;
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_file() {
            count += 1;
        } else if metadata.file_type().is_dir()
            && let Ok(entries) = fs::read_dir(path)
        {
            pending.extend(entries.flatten().map(|entry| entry.path()));
        }
    }
    count
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum WeightFormat {
    SafeTensors,
    PyTorch,
    Gguf,
    Onnx,
    TensorFlow,
    Flax,
}

type WeightArtifact = (WeightFormat, String, Option<(usize, usize)>);

#[derive(Default)]
struct WeightGroup {
    bytes: u64,
    expected_shards: Option<usize>,
    observed_shards: HashSet<usize>,
    inconsistent_shards: bool,
}

impl WeightGroup {
    fn add(&mut self, bytes: u64, shard: Option<(usize, usize)>) {
        self.bytes = self.bytes.saturating_add(bytes);
        if let Some((number, total)) = shard {
            if self
                .expected_shards
                .is_some_and(|expected| expected != total)
            {
                self.inconsistent_shards = true;
            }
            self.expected_shards = Some(total);
            self.observed_shards.insert(number);
        }
    }

    fn complete_bytes(&self) -> Option<u64> {
        if self.inconsistent_shards
            || self
                .expected_shards
                .is_some_and(|expected| self.observed_shards.len() != expected)
        {
            None
        } else {
            Some(self.bytes)
        }
    }
}

fn estimate_model_weight_bytes(model_root: &Path) -> Option<u64> {
    let snapshot = select_snapshot(model_root)?;
    let canonical_model_root = fs::canonicalize(model_root).ok()?;
    let mut seen_files = HashSet::new();
    let mut groups: HashMap<(WeightFormat, PathBuf, String), WeightGroup> = HashMap::new();
    let mut pending = vec![snapshot.clone()];

    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
            continue;
        }
        if !metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let Some((format, group_name, shard)) = weight_artifact(file_name) else {
            continue;
        };
        let Ok(physical_path) = fs::canonicalize(&path) else {
            continue;
        };
        if !physical_path.starts_with(&canonical_model_root)
            || !seen_files.insert(physical_path.clone())
        {
            continue;
        }
        let Ok(physical_metadata) = fs::metadata(&physical_path) else {
            continue;
        };
        if !physical_metadata.is_file() {
            continue;
        }

        let relative_parent = path
            .parent()
            .and_then(|parent| parent.strip_prefix(&snapshot).ok())
            .unwrap_or(Path::new(""))
            .to_path_buf();
        groups
            .entry((format, relative_parent, group_name))
            .or_default()
            .add(physical_metadata.len(), shard);
    }

    // Files in separate component directories (for example Diffusers' UNet
    // and VAE) are additive. Multiple groups in one directory are commonly
    // alternative encodings or quantizations, so retain only the largest.
    let mut per_directory: HashMap<(WeightFormat, PathBuf), u64> = HashMap::new();
    for ((format, directory, _), group) in groups {
        let Some(bytes) = group.complete_bytes() else {
            continue;
        };
        per_directory
            .entry((format, directory))
            .and_modify(|current| *current = (*current).max(bytes))
            .or_insert(bytes);
    }

    let mut per_format: HashMap<WeightFormat, u64> = HashMap::new();
    for ((format, _), bytes) in per_directory {
        let total = per_format.entry(format).or_default();
        *total = total.saturating_add(bytes);
    }
    per_format.into_values().max().filter(|bytes| *bytes > 0)
}

fn select_snapshot(model_root: &Path) -> Option<PathBuf> {
    let snapshots_root = model_root.join("snapshots");
    let snapshots: Vec<_> = fs::read_dir(&snapshots_root)
        .ok()?
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect();
    if snapshots.is_empty() {
        return None;
    }

    let referenced = referenced_snapshots(model_root, &snapshots_root);
    newest_path(if referenced.is_empty() {
        snapshots
    } else {
        referenced
    })
}

fn referenced_snapshots(model_root: &Path, snapshots_root: &Path) -> Vec<PathBuf> {
    let mut referenced = Vec::new();
    let mut pending = vec![model_root.join("refs")];
    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.file_type().is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
        } else if metadata.file_type().is_file()
            && metadata.len() <= MAX_REFERENCE_BYTES
            && let Some(contents) = read_small_text(&path, MAX_REFERENCE_BYTES)
        {
            let revision = contents.trim();
            if safe_snapshot_name(revision) {
                let candidate = snapshots_root.join(revision);
                if candidate.is_dir() {
                    referenced.push(candidate);
                }
            }
        }
    }
    referenced
}

fn read_small_text(path: &Path, limit: u64) -> Option<String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .ok()?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)
        .ok()?;
    (bytes.len() as u64 <= limit)
        .then(|| String::from_utf8(bytes).ok())
        .flatten()
}

fn safe_snapshot_name(value: &str) -> bool {
    !value.is_empty()
        && matches!(
            Path::new(value).components().collect::<Vec<_>>().as_slice(),
            [Component::Normal(_)]
        )
}

fn newest_path(paths: Vec<PathBuf>) -> Option<PathBuf> {
    paths.into_iter().max_by_key(|path| {
        collect_tree_stats(path)
            .last_modified
            .unwrap_or(SystemTime::UNIX_EPOCH)
    })
}

fn inspect_local_metadata(model_root: &Path, snapshot_count: usize) -> LocalModelMetadata {
    let mut result = LocalModelMetadata::default();
    if snapshot_count > 1 {
        result
            .warnings
            .push(CacheWarning::MultipleSnapshots(snapshot_count));
    }

    let Some(snapshot) = select_snapshot(model_root) else {
        return result;
    };
    let commit = snapshot
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or_default()
        .to_owned();
    let references = revision_references(model_root)
        .remove(&commit)
        .unwrap_or_default();
    if references.len() > 1 {
        result
            .warnings
            .push(CacheWarning::DuplicateRevisionReferences {
                commit: commit.clone(),
                references: references.clone(),
            });
    }
    result.selected_revision = Some(SelectedRevision { commit, references });
    inspect_cache_markers(&model_root.join("blobs"), &mut result.warnings);

    let config_path = snapshot.join("config.json");
    let config = read_json(
        &config_path,
        model_root,
        MAX_JSON_BYTES,
        &mut result.warnings,
    );
    if let Some(config) = config.as_ref() {
        result.identity = identity_from_config(config);
        result.maximum_context_length = first_u64(
            config,
            &[
                "max_position_embeddings",
                "n_positions",
                "max_sequence_length",
                "seq_length",
                "context_length",
                "model_max_length",
            ],
        );
        result.pipeline_task = first_string(config, &["pipeline_tag", "task"]);
        result.license = first_string(config, &["license", "license_name"]);
        result.precision.dtype =
            first_string(config, &["torch_dtype", "dtype"]).map(|value| parse_dtype(&value));
        result.transformer = transformer_dimensions(config, result.precision.dtype.as_ref());
        result.requires_remote_code = config
            .get("auto_map")
            .and_then(Value::as_object)
            .is_some_and(|mapping| !mapping.is_empty())
            || config
                .get("trust_remote_code")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        result.precision.quantization = quantization_from_config(config);
        result.parameters = reported_parameter_count(config, MetadataLocation::Config);
    }

    for name in ["README.md", "README.MD", "readme.md"] {
        let path = snapshot.join(name);
        if let Some(front_matter) =
            read_model_card_front_matter(&path, model_root, &mut result.warnings)
        {
            result.pipeline_task = result.pipeline_task.or_else(|| {
                front_matter
                    .get("pipeline_tag")
                    .or_else(|| front_matter.get("task"))
                    .cloned()
            });
            result.license = result
                .license
                .or_else(|| front_matter.get("license").cloned());
            if result.parameters.is_none() {
                result.parameters = [
                    "parameter_count",
                    "parameters",
                    "num_parameters",
                    "n_params",
                ]
                .iter()
                .find_map(|key| front_matter.get(*key))
                .and_then(|value| parse_human_count(value))
                .map(|value| ParameterCount {
                    value,
                    confidence: ParameterCountConfidence::Reported,
                    source: MetadataSource::ModelCard,
                });
            }
            break;
        }
    }

    let artifacts = inspect_snapshot_artifacts(model_root, &snapshot, &mut result.warnings);
    result.weight_formats = artifacts.formats.into_iter().collect();
    if let Some(exact) = artifacts.exact_parameters {
        result.parameters = Some(exact);
    } else if result.parameters.is_none() {
        result.parameters = artifacts.reported_parameters;
    }
    if result.precision.dtype.is_none() {
        result.precision.dtype = artifacts.dtype;
    }
    if result.precision.quantization.is_none() {
        result.precision.quantization = artifacts.quantization;
    }
    if result.transformer.dtype_bytes.is_none() {
        result.transformer.dtype_bytes = result.precision.dtype.as_ref().and_then(dtype_bytes);
    }

    result.completeness = if artifacts.has_weights {
        if result.warnings.iter().any(CacheWarning::indicates_partial) {
            CacheCompleteness::Partial
        } else {
            CacheCompleteness::Complete
        }
    } else if result.warnings.iter().any(CacheWarning::indicates_partial) {
        CacheCompleteness::Partial
    } else {
        CacheCompleteness::Unknown
    };
    result
}

fn inspect_cache_markers(root: &Path, warnings: &mut Vec<CacheWarning>) {
    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".incomplete") {
            warnings.push(CacheWarning::IncompleteArtifact(path));
        } else if lower.ends_with(".lock") {
            warnings.push(CacheWarning::LockArtifact(path));
        }
    }
}

impl CacheWarning {
    fn indicates_partial(&self) -> bool {
        matches!(
            self,
            Self::MissingReferencedFile(_)
                | Self::IncompleteArtifact(_)
                | Self::LockArtifact(_)
                | Self::IncompleteShardSet { .. }
        )
    }
}

fn revision_references(model_root: &Path) -> HashMap<String, Vec<String>> {
    let refs_root = model_root.join("refs");
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    let mut pending = vec![refs_root.clone()];
    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
            continue;
        }
        if !metadata.is_file() || metadata.len() > MAX_REFERENCE_BYTES {
            continue;
        }
        let Some(value) = read_small_text(&path, MAX_REFERENCE_BYTES) else {
            continue;
        };
        let commit = value.trim();
        if commit.is_empty() || commit.contains(['/', '\\']) {
            continue;
        }
        let reference = path
            .strip_prefix(&refs_root)
            .ok()
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default();
        result.entry(commit.to_owned()).or_default().push(reference);
    }
    for references in result.values_mut() {
        references.sort();
        references.dedup();
    }
    result
}

enum MetadataLocation {
    Config,
    SafeTensors,
}

fn reported_parameter_count(value: &Value, location: MetadataLocation) -> Option<ParameterCount> {
    const KEYS: [&str; 6] = [
        "parameter_count",
        "parameters",
        "num_parameters",
        "n_parameters",
        "n_params",
        "total_parameters",
    ];
    for key in KEYS {
        let Some(raw) = value.get(key) else {
            continue;
        };
        let count = json_u64(raw).or_else(|| raw.as_str().and_then(parse_human_count));
        if let Some(value) = count.filter(|value| *value > 0) {
            let source = match location {
                MetadataLocation::Config => MetadataSource::ConfigField(key.to_owned()),
                MetadataLocation::SafeTensors => MetadataSource::SafeTensorsMetadata,
            };
            return Some(ParameterCount {
                value,
                confidence: ParameterCountConfidence::Reported,
                source,
            });
        }
    }
    None
}

fn parse_human_count(value: &str) -> Option<u64> {
    let normalized = value.trim().to_ascii_lowercase().replace(['_', ','], "");
    let (number, multiplier) = match normalized.as_bytes().last().copied() {
        Some(b'k') => (&normalized[..normalized.len() - 1], 1_000_f64),
        Some(b'm') => (&normalized[..normalized.len() - 1], 1_000_000_f64),
        Some(b'b') => (&normalized[..normalized.len() - 1], 1_000_000_000_f64),
        _ => return normalized.parse().ok(),
    };
    let number: f64 = number.parse().ok()?;
    let result = number * multiplier;
    (result.is_finite() && result > 0.0 && result <= u64::MAX as f64)
        .then_some(result.round() as u64)
}

fn identity_from_config(config: &Value) -> ModelIdentity {
    let architectures: Vec<String> = config
        .get("architectures")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let model_type = first_string(config, &["model_type"]);
    let family = first_string(config, &["model_family", "family"])
        .or_else(|| model_type.clone())
        .or_else(|| {
            architectures
                .first()
                .map(|value| architecture_family(value))
        });
    ModelIdentity {
        architectures,
        model_type,
        family,
    }
}

fn architecture_family(architecture: &str) -> String {
    architecture
        .trim_end_matches("ForCausalLM")
        .trim_end_matches("ForConditionalGeneration")
        .trim_end_matches("ForSequenceClassification")
        .trim_end_matches("Model")
        .to_owned()
}

fn first_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn first_u64(value: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| value.get(key).and_then(json_u64))
        .filter(|value| *value > 0)
}

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| value.try_into().ok()))
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

fn parse_dtype(value: &str) -> ModelDtype {
    let normalized = value.trim().to_ascii_lowercase().replace(['_', '-'], "");
    match normalized.as_str() {
        "float64" | "f64" => ModelDtype::Float64,
        "float32" | "f32" | "fp32" => ModelDtype::Float32,
        "float16" | "f16" | "fp16" | "half" => ModelDtype::Float16,
        "bfloat16" | "bf16" => ModelDtype::BFloat16,
        value if value.starts_with("float8") || value.starts_with("fp8") => ModelDtype::Float8,
        "int8" | "i8" => ModelDtype::Int8,
        "int4" | "i4" => ModelDtype::Int4,
        _ => ModelDtype::Other(value.trim().to_owned()),
    }
}

fn transformer_dimensions(config: &Value, dtype: Option<&ModelDtype>) -> TransformerDimensions {
    TransformerDimensions {
        hidden_size: first_u64(config, &["hidden_size", "d_model", "n_embd"]),
        num_hidden_layers: first_u64(
            config,
            &[
                "num_hidden_layers",
                "n_layer",
                "num_layers",
                "decoder_layers",
            ],
        ),
        num_attention_heads: first_u64(
            config,
            &["num_attention_heads", "n_head", "encoder_attention_heads"],
        ),
        num_key_value_heads: first_u64(config, &["num_key_value_heads", "n_head_kv"]),
        head_dim: first_u64(config, &["head_dim", "attention_head_dim"]),
        dtype_bytes: dtype.and_then(dtype_bytes),
    }
}

fn dtype_bytes(dtype: &ModelDtype) -> Option<u8> {
    match dtype {
        ModelDtype::Float64 => Some(8),
        ModelDtype::Float32 => Some(4),
        ModelDtype::Float16 | ModelDtype::BFloat16 => Some(2),
        ModelDtype::Float8 | ModelDtype::Int8 => Some(1),
        // Quantized weight dtypes do not establish the runtime KV-cache dtype.
        ModelDtype::Int4 | ModelDtype::Other(_) => None,
    }
}

fn quantization_from_config(config: &Value) -> Option<QuantizationInfo> {
    let quantization = config.get("quantization_config")?;
    let method = first_string(
        quantization,
        &["quant_method", "quantization_method", "method"],
    )
    .or_else(|| first_string(config, &["quantization_method"]))?;
    let bits = first_u64(quantization, &["bits", "weight_bits"])
        .and_then(|value| u8::try_from(value).ok());
    let variant = first_string(
        quantization,
        &["checkpoint_format", "version", "format", "quant_type"],
    );
    Some(QuantizationInfo {
        method,
        bits,
        variant,
        source: QuantizationSource::Config,
    })
}

/// Reads a regular file resolved under `model_root`, bounded by `limit`.
///
/// Pushes `OversizedMetadata` and returns `None` when the file exceeds `limit`
/// or cannot be read within it.
fn read_bounded_file(
    path: &Path,
    model_root: &Path,
    limit: u64,
    warnings: &mut Vec<CacheWarning>,
) -> Option<Vec<u8>> {
    let resolved = resolve_local_file(path, model_root)?;
    let metadata = fs::metadata(&resolved).ok()?;
    if metadata.len() > limit {
        warnings.push(CacheWarning::OversizedMetadata(path.to_path_buf()));
        return None;
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    let read_result = fs::File::open(resolved)
        .ok()?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes);
    if read_result.is_err() || bytes.len() as u64 > limit {
        warnings.push(CacheWarning::OversizedMetadata(path.to_path_buf()));
        return None;
    }
    Some(bytes)
}

fn read_json(
    path: &Path,
    model_root: &Path,
    limit: u64,
    warnings: &mut Vec<CacheWarning>,
) -> Option<Value> {
    let bytes = read_bounded_file(path, model_root, limit, warnings)?;
    match serde_json::from_slice(&bytes) {
        Ok(value) => Some(value),
        Err(_) => {
            warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
            None
        }
    }
}

fn read_model_card_front_matter(
    path: &Path,
    model_root: &Path,
    warnings: &mut Vec<CacheWarning>,
) -> Option<HashMap<String, String>> {
    let bytes = read_bounded_file(path, model_root, MAX_MODEL_CARD_BYTES, warnings)?;
    let text = match std::str::from_utf8(&bytes) {
        Ok(text) => text,
        Err(_) => {
            warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
            return None;
        }
    };
    let mut lines = text.lines();
    if lines.next().map(str::trim) != Some("---") {
        return None;
    }
    let mut values = HashMap::new();
    for line in lines.take(256) {
        if line.trim() == "---" {
            return Some(values);
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().trim_matches(['"', '\'']);
        if !value.is_empty() {
            values.insert(key.trim().to_ascii_lowercase(), value.to_owned());
        }
    }
    warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
    None
}

fn resolve_local_file(path: &Path, model_root: &Path) -> Option<PathBuf> {
    let root = fs::canonicalize(model_root).ok()?;
    let resolved = fs::canonicalize(path).ok()?;
    (resolved.starts_with(root) && resolved.is_file()).then_some(resolved)
}

#[derive(Default)]
struct SnapshotArtifacts {
    formats: std::collections::BTreeSet<WeightFormat>,
    has_weights: bool,
    exact_parameters: Option<ParameterCount>,
    reported_parameters: Option<ParameterCount>,
    dtype: Option<ModelDtype>,
    quantization: Option<QuantizationInfo>,
}

#[derive(Default)]
struct TensorGroup {
    parameters: u64,
    valid_headers: usize,
    expected_shards: Option<usize>,
    observed_shards: HashSet<usize>,
}

fn inspect_snapshot_artifacts(
    model_root: &Path,
    snapshot: &Path,
    warnings: &mut Vec<CacheWarning>,
) -> SnapshotArtifacts {
    let canonical_root = fs::canonicalize(model_root).ok();
    let mut result = SnapshotArtifacts::default();
    let mut groups: HashMap<(PathBuf, String), TensorGroup> = HashMap::new();
    let mut dtypes = HashSet::new();
    let mut pending = vec![snapshot.to_path_buf()];

    while let Some(path) = pending.pop() {
        let Ok(metadata) = fs::symlink_metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if let Ok(entries) = fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
            continue;
        }
        let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let lower = file_name.to_ascii_lowercase();
        if lower.ends_with(".incomplete") {
            warnings.push(CacheWarning::IncompleteArtifact(path.clone()));
            continue;
        }
        if lower.ends_with(".lock") {
            warnings.push(CacheWarning::LockArtifact(path.clone()));
            continue;
        }
        if lower.ends_with(".index.json") {
            inspect_index_manifest(model_root, snapshot, &path, warnings);
            continue;
        }
        let Some(format) = detect_weight_format(&lower) else {
            continue;
        };
        let Ok(physical_path) = fs::canonicalize(&path) else {
            continue;
        };
        if canonical_root
            .as_ref()
            .is_some_and(|root| !physical_path.starts_with(root))
            || !physical_path.is_file()
        {
            continue;
        }
        result.formats.insert(format);
        result.has_weights = true;

        if format == WeightFormat::Gguf && result.quantization.is_none() {
            result.quantization = quantization_from_gguf_name(&lower);
        }
        if format != WeightFormat::SafeTensors {
            continue;
        }
        let Some(header) = read_safetensors_header(&physical_path, warnings) else {
            continue;
        };
        if result.reported_parameters.is_none() {
            result.reported_parameters = header.get("__metadata__").and_then(|metadata| {
                reported_parameter_count(metadata, MetadataLocation::SafeTensors)
            });
        }
        let Some(parameters) = safetensors_parameter_count(&header) else {
            continue;
        };
        collect_safetensors_dtypes(&header, &mut dtypes);
        let relative_parent = path
            .parent()
            .and_then(|parent| parent.strip_prefix(snapshot).ok())
            .unwrap_or(Path::new(""))
            .to_path_buf();
        let (group_name, shard) = lower
            .strip_suffix(".safetensors")
            .map(shard_group)
            .unwrap_or((&lower, None));
        let group = groups
            .entry((relative_parent, group_name.to_owned()))
            .or_default();
        group.parameters = group.parameters.saturating_add(parameters);
        group.valid_headers += 1;
        if let Some((number, total)) = shard {
            group.expected_shards = Some(total);
            group.observed_shards.insert(number);
        }
    }

    let mut per_directory: HashMap<PathBuf, u64> = HashMap::new();
    for ((directory, group_name), group) in groups {
        if let Some(expected) = group.expected_shards
            && group.observed_shards.len() != expected
        {
            warnings.push(CacheWarning::IncompleteShardSet {
                group: group_name,
                expected,
                found: group.observed_shards.len(),
            });
            continue;
        }
        if group.valid_headers > 0 {
            per_directory
                .entry(directory)
                .and_modify(|current| *current = (*current).max(group.parameters))
                .or_insert(group.parameters);
        }
    }
    let exact = per_directory
        .into_values()
        .try_fold(0_u64, u64::checked_add)
        .filter(|value| *value > 0);
    result.exact_parameters = exact.map(|value| ParameterCount {
        value,
        confidence: ParameterCountConfidence::Exact,
        source: MetadataSource::SafeTensorsHeader,
    });
    if dtypes.len() == 1 {
        result.dtype = dtypes.into_iter().next();
    }
    result
}

/// Recognized weight-file extensions and the format each one denotes.
const WEIGHT_EXTENSIONS: &[(&str, WeightFormat)] = &[
    (".safetensors", WeightFormat::SafeTensors),
    (".gguf", WeightFormat::Gguf),
    (".bin", WeightFormat::PyTorch),
    (".pth", WeightFormat::PyTorch),
    (".pt", WeightFormat::PyTorch),
    (".onnx", WeightFormat::Onnx),
    (".h5", WeightFormat::TensorFlow),
    (".ckpt", WeightFormat::TensorFlow),
    (".msgpack", WeightFormat::Flax),
];

/// Match a lowercased file name to its weight format and the extension that
/// matched, rejecting training artifacts such as optimizer and scheduler state.
fn weight_format(file_name: &str) -> Option<(&'static str, WeightFormat)> {
    if is_non_model_artifact(file_name) {
        return None;
    }
    WEIGHT_EXTENSIONS
        .iter()
        .copied()
        .find(|(extension, _)| file_name.ends_with(extension))
}

fn detect_weight_format(file_name: &str) -> Option<WeightFormat> {
    weight_format(file_name).map(|(_, format)| format)
}

fn inspect_index_manifest(
    model_root: &Path,
    snapshot: &Path,
    path: &Path,
    warnings: &mut Vec<CacheWarning>,
) {
    let Some(index) = read_json(path, model_root, MAX_JSON_BYTES, warnings) else {
        return;
    };
    let Some(weight_map) = index.get("weight_map").and_then(Value::as_object) else {
        return;
    };
    let canonical_root = fs::canonicalize(model_root).ok();
    let mut files = HashSet::new();
    for file in weight_map.values().filter_map(Value::as_str) {
        if !safe_relative_path(file) || !files.insert(file) {
            continue;
        }
        let candidate = snapshot.join(file);
        let present = fs::canonicalize(&candidate).ok().is_some_and(|resolved| {
            canonical_root
                .as_ref()
                .is_none_or(|root| resolved.starts_with(root))
                && resolved.is_file()
        });
        if !present {
            warnings.push(CacheWarning::MissingReferencedFile(PathBuf::from(file)));
        }
    }
}

fn safe_relative_path(value: &str) -> bool {
    let path = Path::new(value);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn read_safetensors_header(path: &Path, warnings: &mut Vec<CacheWarning>) -> Option<Value> {
    let mut file = fs::File::open(path).ok()?;
    let mut length_bytes = [0_u8; 8];
    if file.read_exact(&mut length_bytes).is_err() {
        warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
        return None;
    }
    let length = u64::from_le_bytes(length_bytes);
    let file_length = file.metadata().ok()?.len();
    if length == 0 || length > MAX_WEIGHT_HEADER_BYTES || length > file_length.saturating_sub(8) {
        warnings.push(if length > MAX_WEIGHT_HEADER_BYTES {
            CacheWarning::OversizedMetadata(path.to_path_buf())
        } else {
            CacheWarning::MalformedMetadata(path.to_path_buf())
        });
        return None;
    }
    let mut bytes = vec![0; length as usize];
    if file.seek(SeekFrom::Start(8)).is_err() || file.read_exact(&mut bytes).is_err() {
        warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
        return None;
    }
    match serde_json::from_slice(&bytes) {
        Ok(value) => Some(value),
        Err(_) => {
            warnings.push(CacheWarning::MalformedMetadata(path.to_path_buf()));
            None
        }
    }
}

fn safetensors_parameter_count(header: &Value) -> Option<u64> {
    let object = header.as_object()?;
    let mut total = 0_u64;
    let mut tensors = 0_usize;
    for (name, tensor) in object {
        if name == "__metadata__" {
            continue;
        }
        let shape = tensor.get("shape")?.as_array()?;
        let count = shape
            .iter()
            .map(json_u64)
            .try_fold(1_u64, |total, value| total.checked_mul(value?))?;
        total = total.checked_add(count)?;
        tensors += 1;
    }
    (tensors > 0).then_some(total)
}

fn collect_safetensors_dtypes(header: &Value, dtypes: &mut HashSet<ModelDtype>) {
    let Some(object) = header.as_object() else {
        return;
    };
    for (name, tensor) in object {
        if name != "__metadata__"
            && let Some(dtype) = tensor.get("dtype").and_then(Value::as_str)
        {
            dtypes.insert(parse_dtype(dtype));
        }
    }
}

fn quantization_from_gguf_name(file_name: &str) -> Option<QuantizationInfo> {
    let stem = file_name.strip_suffix(".gguf")?;
    let variant = stem
        .split(['-', '.'])
        .rev()
        .find(|part| {
            let lower = part.to_ascii_lowercase();
            lower.starts_with('q') && lower.chars().any(|character| character.is_ascii_digit())
        })?
        .to_ascii_uppercase();
    let bits = variant
        .chars()
        .skip_while(|character| !character.is_ascii_digit())
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok();
    Some(QuantizationInfo {
        method: "GGUF".to_owned(),
        bits,
        variant: Some(variant),
        source: QuantizationSource::FileName,
    })
}

fn weight_artifact(file_name: &str) -> Option<WeightArtifact> {
    let lower = file_name.to_ascii_lowercase();
    let (extension, format) = weight_format(&lower)?;
    let stem = lower.strip_suffix(extension)?;
    let (group, shard) = shard_group(stem);
    Some((format, group.to_owned(), shard))
}

fn is_non_model_artifact(file_name: &str) -> bool {
    [
        "optimizer",
        "scheduler",
        "training_args",
        "trainer_state",
        "rng_state",
        "scaler",
    ]
    .iter()
    .any(|marker| file_name.contains(marker))
}

fn shard_group(stem: &str) -> (&str, Option<(usize, usize)>) {
    let Some((before_of, total)) = stem.rsplit_once("-of-") else {
        return (stem, None);
    };
    if total.is_empty() || !total.bytes().all(|byte| byte.is_ascii_digit()) {
        return (stem, None);
    }
    let Some((base, shard)) = before_of.rsplit_once('-') else {
        return (stem, None);
    };
    if shard.is_empty() || !shard.bytes().all(|byte| byte.is_ascii_digit()) {
        return (stem, None);
    }
    let Ok(number) = shard.parse() else {
        return (stem, None);
    };
    let Ok(total) = total.parse() else {
        return (stem, None);
    };
    if number == 0 || total == 0 || number > total {
        return (stem, None);
    }
    (base, Some((number, total)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::HashMap,
        fs::File,
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT_TEST_DIR: AtomicU64 = AtomicU64::new(0);

    fn decoder_metadata() -> LocalModelMetadata {
        LocalModelMetadata {
            identity: ModelIdentity {
                architectures: vec!["LlamaForCausalLM".into()],
                model_type: Some("llama".into()),
                family: Some("llama".into()),
            },
            precision: PrecisionInfo {
                dtype: Some(ModelDtype::Float16),
                quantization: None,
            },
            pipeline_task: Some("text-generation".into()),
            completeness: CacheCompleteness::Complete,
            weight_formats: vec![WeightFormat::SafeTensors],
            transformer: TransformerDimensions {
                hidden_size: Some(4_096),
                num_hidden_layers: Some(32),
                num_attention_heads: Some(32),
                num_key_value_heads: Some(8),
                head_dim: Some(128),
                dtype_bytes: Some(2),
            },
            ..LocalModelMetadata::default()
        }
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let sequence = NEXT_TEST_DIR.fetch_add(1, Ordering::Relaxed);
            let path = env::current_dir()
                .expect("current directory")
                .join("target")
                .join("cache-tests")
                .join(format!("{label}-{}-{sequence}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).expect("create test directory");
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cache_root_environment_precedence_and_fallbacks() {
        let cases = [
            (
                [("HF_HUB_CACHE", "/hub"), ("HF_HOME", "/home")].as_slice(),
                PathBuf::from("/hub"),
            ),
            (
                [("HF_HOME", "/hf"), ("XDG_CACHE_HOME", "/xdg")].as_slice(),
                PathBuf::from("/hf/hub"),
            ),
            (
                [("XDG_CACHE_HOME", "/xdg"), ("HOME", "/user")].as_slice(),
                PathBuf::from("/xdg/huggingface/hub"),
            ),
            (
                [("HF_HUB_CACHE", ""), ("HOME", "/user")].as_slice(),
                PathBuf::from("/user/.cache/huggingface/hub"),
            ),
        ];

        for (variables, expected) in cases {
            let variables: HashMap<_, _> = variables.iter().copied().collect();
            assert_eq!(
                cache_root_from(|name| variables.get(name).map(|value| OsString::from(*value))),
                Some(expected)
            );
        }
        assert_eq!(cache_root_from(|_| None), None);
    }

    #[test]
    fn grouped_query_attention_kv_math_is_exact() {
        let metadata = decoder_metadata();
        let estimate = metadata.context_vram_estimate(Some(10 * BYTES_PER_MIB), 4_096, 20);
        assert_eq!(estimate.kv_cache_bytes, Some(2 * 4_096 * 32 * 8 * 128 * 2));
        assert_eq!(estimate.base_allocation_bytes, Some(12 * BYTES_PER_MIB));
        assert_eq!(
            estimate.allocation_input_bytes,
            Some(12 * BYTES_PER_MIB + 2 * 4_096 * 32 * 8 * 128 * 2)
        );
        assert_eq!(estimate.confidence, EstimateConfidence::Exact);
    }

    #[test]
    fn runtime_overhead_is_applied_once_to_exact_weight_bytes() {
        let metadata = decoder_metadata();
        let estimate = metadata.context_vram_estimate(Some(101), 1, 20);

        assert_eq!(estimate.base_allocation_bytes, Some(122));
        assert_eq!(
            estimate.allocation_input_bytes,
            estimate
                .kv_cache_bytes
                .and_then(|kv| 122_u64.checked_add(kv))
        );
    }

    #[test]
    fn missing_kv_heads_uses_mha_and_derived_head_dimension() {
        let mut metadata = decoder_metadata();
        metadata.transformer.num_key_value_heads = None;
        metadata.transformer.head_dim = None;
        let estimate = metadata.context_vram_estimate(Some(BYTES_PER_MIB), 2_048, 0);
        assert_eq!(estimate.kv_cache_bytes, Some(2 * 2_048 * 32 * 32 * 128 * 2));
        assert_eq!(estimate.confidence, EstimateConfidence::Estimated);
    }

    #[test]
    fn context_is_capped_at_model_maximum() {
        let mut metadata = decoder_metadata();
        metadata.maximum_context_length = Some(4_096);
        let estimate = metadata.context_vram_estimate(Some(BYTES_PER_MIB), 8_192, 20);
        assert_eq!(estimate.effective_tokens, 4_096);
        assert!(estimate.capped_to_model_max);
        assert_eq!(estimate.confidence, EstimateConfidence::Exact);
        assert!(estimate.reason.as_deref().unwrap().contains("capped"));
    }

    #[test]
    fn unknown_dimensions_are_unavailable() {
        let mut metadata = decoder_metadata();
        metadata.transformer.num_hidden_layers = None;
        let estimate = metadata.context_vram_estimate(Some(BYTES_PER_MIB), 4_096, 20);
        assert_eq!(estimate.confidence, EstimateConfidence::Unavailable);
        assert!(estimate.allocation_input_bytes.is_none());
    }

    #[test]
    fn encoder_and_overflow_do_not_invent_kv_sizes() {
        let mut encoder = decoder_metadata();
        encoder.pipeline_task = Some("feature-extraction".into());
        encoder.identity.architectures = vec!["BertModel".into()];
        assert_eq!(
            encoder
                .context_vram_estimate(Some(BYTES_PER_MIB), 2_048, 20)
                .confidence,
            EstimateConfidence::Unavailable
        );

        let mut overflow = decoder_metadata();
        overflow.transformer.num_hidden_layers = Some(u64::MAX);
        assert_eq!(
            overflow
                .context_vram_estimate(Some(BYTES_PER_MIB), u64::MAX, 20)
                .confidence,
            EstimateConfidence::Unavailable
        );
    }

    #[test]
    fn context_presets_are_bounded_and_deduplicated() {
        assert_eq!(context_presets(Some(8_192)), [2_048, 4_096, 8_192]);
        assert_eq!(context_presets(Some(6_000)), [2_048, 4_096, 6_000]);
        assert_eq!(context_presets(Some(1_000)), [1_000]);
        assert_eq!(context_presets(None), [2_048, 4_096, 8_192, 16_384, 32_768]);
    }

    #[test]
    fn bfloat16_compute_capability_boundary_is_enforced() {
        let mut metadata = decoder_metadata();
        metadata.precision.dtype = Some(ModelDtype::BFloat16);
        let old = crate::gpu::parse_nvidia_smi_output("old, 100, 100, 7.5")
            .unwrap()
            .unwrap();
        assert_eq!(
            metadata
                .runtime_compatibility(&GpuDetection::Detected(old))
                .status,
            CompatibilityStatus::Unsupported
        );
        let ampere = crate::gpu::parse_nvidia_smi_output("ampere, 100, 100, 8.0")
            .unwrap()
            .unwrap();
        assert_ne!(
            metadata
                .runtime_compatibility(&GpuDetection::Detected(ampere))
                .status,
            CompatibilityStatus::Unsupported
        );

        let mixed =
            crate::gpu::parse_nvidia_smi_output("ampere, 100, 100, 8.0\nolder, 100, 100, 7.5")
                .unwrap()
                .unwrap();
        let assessment = metadata.runtime_compatibility(&GpuDetection::Detected(mixed));
        assert_eq!(assessment.status, CompatibilityStatus::Warning);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| issue.reason.contains("mixed"))
        );
    }

    #[test]
    fn compatibility_reports_incomplete_gguf_and_backend_quantization() {
        let mut metadata = decoder_metadata();
        metadata.completeness = CacheCompleteness::Partial;
        metadata.weight_formats = vec![WeightFormat::Gguf];
        metadata.precision.quantization = Some(QuantizationInfo {
            method: "GPTQ".into(),
            bits: Some(4),
            variant: None,
            source: QuantizationSource::Config,
        });
        let assessment = metadata.runtime_compatibility(&GpuDetection::NoGpu);
        assert_eq!(assessment.status, CompatibilityStatus::Unsupported);
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| issue.reason.contains("incomplete"))
        );
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| issue.reason.contains("llama.cpp"))
        );
        assert!(
            assessment
                .issues
                .iter()
                .any(|issue| issue.reason.contains("backend support"))
        );
    }

    #[test]
    fn scans_models_and_skips_malformed_entries() {
        let root = TestDir::new("scan");
        let model = root.0.join("models--acme--tiny-model");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::create_dir_all(model.join("snapshots/rev-a")).unwrap();
        fs::create_dir_all(model.join("snapshots/rev-b")).unwrap();
        fs::create_dir_all(model.join("refs/pr")).unwrap();
        write_bytes(&model.join("blobs/abc"), b"12345");
        write_bytes(&model.join("refs/main"), b"rev-a");
        write_bytes(&model.join("refs/pr/1"), b"rev-b");
        fs::create_dir_all(root.0.join("datasets--acme--ignored")).unwrap();
        fs::create_dir_all(root.0.join("models--malformed")).unwrap();
        write_bytes(&root.0.join("models--acme--not-a-directory"), b"x");

        let models = scan_models(&root.0).unwrap();
        assert_eq!(models.len(), 1);
        let info = &models[0];
        assert_eq!(info.id, "acme/tiny-model");
        assert_eq!(info.organization, "acme");
        assert_eq!(info.name, "tiny-model");
        assert_eq!(info.snapshot_count, 2);
        assert_eq!(info.revision_count, 2);
        assert_eq!(info.size_bytes, 15);
        assert_eq!(info.estimated_model_weight_bytes, None);
        assert!(info.last_modified.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn snapshot_symlinks_are_not_double_counted_and_broken_links_are_safe() {
        use std::os::unix::fs::symlink;

        let root = TestDir::new("symlinks");
        let model = root.0.join("models--org--model");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::create_dir_all(model.join("snapshots/revision")).unwrap();
        write_bytes(&model.join("blobs/content"), b"content");
        symlink(
            "../../blobs/content",
            model.join("snapshots/revision/weights.bin"),
        )
        .unwrap();
        symlink(
            "../../blobs/missing",
            model.join("snapshots/revision/broken.bin"),
        )
        .unwrap();

        let models = scan_models(&root.0).unwrap();
        assert_eq!(models[0].size_bytes, 7);
        assert_eq!(models[0].estimated_model_weight_bytes, Some(7));
        assert_eq!(models[0].snapshot_count, 1);
    }

    #[test]
    fn estimates_shards_without_adding_alternative_weight_formats() {
        let root = TestDir::new("weight-formats");
        let model = root.0.join("models--org--model");
        let snapshot = model.join("snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        write_bytes(&snapshot.join("model-00001-of-00002.safetensors"), &[0; 7]);
        write_bytes(&snapshot.join("model-00002-of-00002.safetensors"), &[0; 8]);
        write_bytes(&snapshot.join("pytorch_model.bin"), &[0; 12]);
        write_bytes(&snapshot.join("model-q4.gguf"), &[0; 9]);
        write_bytes(&snapshot.join("model-q8.gguf"), &[0; 14]);
        write_bytes(&snapshot.join("optimizer.pt"), &[0; 100]);
        write_bytes(&snapshot.join("config.json"), &[0; 200]);

        let models = scan_models(&root.0).unwrap();
        assert_eq!(models[0].estimated_model_weight_bytes, Some(15));
    }

    #[test]
    fn estimates_bytes_for_every_recognized_weight_format() {
        let root = TestDir::new("all-formats");
        let snapshot = root.0.join("models--org--model/snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        for (file_name, bytes) in [
            ("model.safetensors", 11),
            ("model.bin", 12),
            ("model.gguf", 13),
            ("model.onnx", 14),
            ("model.h5", 15),
            ("model.msgpack", 16),
        ] {
            write_bytes(&snapshot.join(file_name), &vec![0; bytes]);
        }

        let model = &scan_models(&root.0).unwrap()[0];
        // Every recognized format is both listed and sized, so a model cannot
        // be reported as e.g. Onnx while also claiming it has no weight size.
        assert_eq!(
            model.local_metadata().weight_formats,
            [
                WeightFormat::SafeTensors,
                WeightFormat::PyTorch,
                WeightFormat::Gguf,
                WeightFormat::Onnx,
                WeightFormat::TensorFlow,
                WeightFormat::Flax,
            ]
        );
        assert_eq!(model.estimated_model_weight_bytes, Some(16));
    }

    #[test]
    fn adds_weight_components_from_separate_snapshot_directories() {
        let root = TestDir::new("components");
        let model = root.0.join("models--org--diffuser");
        let snapshot = model.join("snapshots/revision");
        fs::create_dir_all(snapshot.join("unet")).unwrap();
        fs::create_dir_all(snapshot.join("vae")).unwrap();
        write_bytes(
            &snapshot.join("unet/diffusion_pytorch_model.safetensors"),
            &[0; 11],
        );
        write_bytes(
            &snapshot.join("vae/diffusion_pytorch_model.safetensors"),
            &[0; 5],
        );

        let models = scan_models(&root.0).unwrap();
        assert_eq!(models[0].estimated_model_weight_bytes, Some(16));
    }

    #[test]
    fn prefers_a_valid_referenced_snapshot_over_unreferenced_snapshots() {
        let root = TestDir::new("referenced-snapshot");
        let model = root.0.join("models--org--model");
        fs::create_dir_all(model.join("snapshots/referenced")).unwrap();
        fs::create_dir_all(model.join("snapshots/unreferenced")).unwrap();
        fs::create_dir_all(model.join("refs")).unwrap();
        write_bytes(
            &model.join("snapshots/referenced/model.safetensors"),
            &[0; 6],
        );
        write_bytes(
            &model.join("snapshots/unreferenced/model.safetensors"),
            &[0; 20],
        );
        write_bytes(&model.join("refs/main"), b"referenced\n");

        let models = scan_models(&root.0).unwrap();
        assert_eq!(models[0].estimated_model_weight_bytes, Some(6));
    }

    #[test]
    fn ignores_oversized_and_unsafe_snapshot_references() {
        let root = TestDir::new("unsafe-snapshot-refs");
        let model = root.0.join("models--org--model");
        let safe = model.join("snapshots/safe");
        fs::create_dir_all(model.join("refs")).unwrap();
        fs::create_dir_all(&safe).unwrap();
        write_bytes(&safe.join("model.safetensors"), &[0; 6]);
        write_bytes(&model.join("refs/traversal"), b"../safe");
        let oversized = File::create(model.join("refs/oversized")).unwrap();
        oversized.set_len(4097).unwrap();

        let models = scan_models(&root.0).unwrap();

        assert_eq!(models[0].estimated_model_weight_bytes, Some(6));
        assert!(
            models[0]
                .local_metadata()
                .selected_revision
                .as_ref()
                .is_some_and(|revision| revision.references.is_empty())
        );
    }

    #[test]
    fn partial_and_metadata_only_snapshots_are_tolerated() {
        let root = TestDir::new("partial");
        let partial = root.0.join("models--org--partial/snapshots/revision");
        let metadata = root.0.join("models--org--metadata/snapshots/revision");
        fs::create_dir_all(&partial).unwrap();
        fs::create_dir_all(&metadata).unwrap();
        write_bytes(&partial.join("model-00001-of-00003.safetensors"), &[0; 4]);
        write_bytes(&metadata.join("config.json"), b"{}");

        let models = scan_models(&root.0).unwrap();
        let estimates: HashMap<_, _> = models
            .iter()
            .map(|model| (model.id.as_str(), model.estimated_model_weight_bytes))
            .collect();
        assert_eq!(estimates["org/partial"], None);
        assert_eq!(estimates["org/metadata"], None);
    }

    #[test]
    fn extracts_config_card_and_exact_safetensors_metadata() {
        let root = TestDir::new("local-metadata");
        let model = root.0.join("models--org--model");
        let snapshot = model.join("snapshots/abc123");
        fs::create_dir_all(model.join("refs")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        write_bytes(&model.join("refs/main"), b"abc123\n");
        write_bytes(
            &snapshot.join("config.json"),
            br#"{
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_parameters": 7000000000,
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 8192,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "pipeline_tag": "text-generation"
            }"#,
        );
        write_bytes(
            &snapshot.join("README.md"),
            b"---\nlicense: apache-2.0\npipeline_tag: ignored-config-wins\n---\n# Model\n",
        );
        write_safetensors(
            &snapshot.join("model.safetensors"),
            br#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"BF16","shape":[4],"data_offsets":[12,20]}}"#,
        );

        let info = scan_models(&root.0).unwrap().remove(0);
        let metadata = info.local_metadata();
        assert_eq!(metadata.identity.architectures, ["LlamaForCausalLM"]);
        assert_eq!(metadata.identity.model_type.as_deref(), Some("llama"));
        assert_eq!(metadata.identity.family.as_deref(), Some("llama"));
        assert_eq!(
            metadata.parameters,
            Some(ParameterCount {
                value: 10,
                confidence: ParameterCountConfidence::Exact,
                source: MetadataSource::SafeTensorsHeader,
            })
        );
        assert_eq!(metadata.precision.dtype, Some(ModelDtype::BFloat16));
        assert_eq!(metadata.maximum_context_length, Some(8192));
        assert_eq!(
            metadata.transformer,
            TransformerDimensions {
                hidden_size: Some(4096),
                num_hidden_layers: Some(32),
                num_attention_heads: Some(32),
                num_key_value_heads: Some(8),
                head_dim: Some(128),
                dtype_bytes: Some(2),
            }
        );
        assert_eq!(metadata.pipeline_task.as_deref(), Some("text-generation"));
        assert_eq!(metadata.license.as_deref(), Some("apache-2.0"));
        assert_eq!(metadata.weight_formats, [WeightFormat::SafeTensors]);
        assert_eq!(
            metadata.selected_revision,
            Some(SelectedRevision {
                commit: "abc123".into(),
                references: vec!["main".into()],
            })
        );
        assert_eq!(metadata.completeness, CacheCompleteness::Complete);
    }

    #[test]
    fn reports_quantization_formats_and_reported_parameter_fallback() {
        let root = TestDir::new("quantization");
        let model = root.0.join("models--org--quantized");
        let snapshot = model.join("snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        write_bytes(
            &snapshot.join("config.json"),
            br#"{
                "parameter_count": "1.5B",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                    "checkpoint_format": "gptq_v2"
                }
            }"#,
        );
        write_bytes(&snapshot.join("model-Q5_K_M.gguf"), b"GGUF");
        write_bytes(&snapshot.join("model.onnx"), b"onnx");
        write_bytes(&snapshot.join("flax_model.msgpack"), b"flax");

        let metadata = scan_models(&root.0).unwrap()[0].local_metadata();
        assert_eq!(
            metadata.parameters,
            Some(ParameterCount {
                value: 1_500_000_000,
                confidence: ParameterCountConfidence::Reported,
                source: MetadataSource::ConfigField("parameter_count".into()),
            })
        );
        assert_eq!(
            metadata.precision.quantization,
            Some(QuantizationInfo {
                method: "gptq".into(),
                bits: Some(4),
                variant: Some("gptq_v2".into()),
                source: QuantizationSource::Config,
            })
        );
        assert_eq!(
            metadata.weight_formats,
            [WeightFormat::Gguf, WeightFormat::Onnx, WeightFormat::Flax]
        );
    }

    #[test]
    fn identifies_missing_index_shards_and_download_markers() {
        let root = TestDir::new("incomplete-metadata");
        let model = root.0.join("models--org--partial");
        let snapshot = model.join("snapshots/revision");
        fs::create_dir_all(model.join("blobs")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        write_bytes(
            &snapshot.join("model.safetensors.index.json"),
            br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        );
        write_safetensors(
            &snapshot.join("model-00001-of-00002.safetensors"),
            br#"{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
        );
        write_bytes(&model.join("blobs/pending.incomplete"), b"");
        write_bytes(&model.join("blobs/download.lock"), b"");

        let metadata = scan_models(&root.0).unwrap()[0].local_metadata();
        assert_eq!(metadata.completeness, CacheCompleteness::Partial);
        assert!(metadata.warnings.iter().any(|warning| matches!(
            warning,
            CacheWarning::MissingReferencedFile(path)
                if path == Path::new("model-00002-of-00002.safetensors")
        )));
        assert!(metadata.warnings.iter().any(|warning| matches!(
            warning,
            CacheWarning::IncompleteShardSet {
                expected: 2,
                found: 1,
                ..
            }
        )));
        assert!(
            metadata
                .warnings
                .iter()
                .any(|warning| matches!(warning, CacheWarning::IncompleteArtifact(_)))
        );
        assert!(
            metadata
                .warnings
                .iter()
                .any(|warning| matches!(warning, CacheWarning::LockArtifact(_)))
        );
        assert_eq!(metadata.parameters, None);
    }

    #[test]
    fn warns_about_multiple_snapshots_and_duplicate_refs() {
        let root = TestDir::new("duplicate-revisions");
        let model = root.0.join("models--org--model");
        fs::create_dir_all(model.join("snapshots/selected")).unwrap();
        fs::create_dir_all(model.join("snapshots/other")).unwrap();
        fs::create_dir_all(model.join("refs/pr")).unwrap();
        write_bytes(&model.join("refs/main"), b"selected");
        write_bytes(&model.join("refs/pr/1"), b"selected");
        write_bytes(&model.join("snapshots/selected/model.gguf"), b"GGUF");

        let metadata = scan_models(&root.0).unwrap()[0].local_metadata();
        assert_eq!(
            metadata.selected_revision.as_ref().unwrap().references,
            ["main", "pr/1"]
        );
        assert!(
            metadata
                .warnings
                .contains(&CacheWarning::MultipleSnapshots(2))
        );
        assert!(metadata.warnings.iter().any(|warning| matches!(
            warning,
            CacheWarning::DuplicateRevisionReferences { commit, references }
                if commit == "selected" && references == &["main", "pr/1"]
        )));
    }

    #[test]
    fn malformed_and_oversized_metadata_are_bounded_and_nonfatal() {
        let root = TestDir::new("bounded-metadata");
        let malformed = root.0.join("models--org--malformed/snapshots/revision");
        let oversized = root.0.join("models--org--oversized/snapshots/revision");
        fs::create_dir_all(&malformed).unwrap();
        fs::create_dir_all(&oversized).unwrap();
        write_bytes(&malformed.join("config.json"), b"{not json");
        write_bytes(&malformed.join("README.md"), b"\xff\xfe");
        write_bytes(&malformed.join("model-Q4_K_M.gguf"), b"GGUF");
        let file = File::create(oversized.join("config.json")).unwrap();
        file.set_len(MAX_JSON_BYTES + 1).unwrap();
        write_bytes(&oversized.join("pytorch_model.bin"), b"weights");

        let models = scan_models(&root.0).unwrap();
        for model in models {
            let metadata = model.local_metadata();
            assert!(metadata.parameters.is_none());
            assert!(metadata.identity.architectures.is_empty());
            assert!(metadata.warnings.iter().any(|warning| matches!(
                warning,
                CacheWarning::MalformedMetadata(_) | CacheWarning::OversizedMetadata(_)
            )));
            assert_ne!(metadata.completeness, CacheCompleteness::Partial);
        }
    }

    #[test]
    fn uses_newest_snapshot_when_no_reference_is_valid() {
        let root = TestDir::new("newest-fallback");
        let model = root.0.join("models--org--model");
        let old = model.join("snapshots/old");
        let newest = model.join("snapshots/newest");
        fs::create_dir_all(model.join("refs")).unwrap();
        fs::create_dir_all(&old).unwrap();
        write_bytes(&old.join("config.json"), br#"{"model_type":"old"}"#);
        std::thread::sleep(std::time::Duration::from_millis(10));
        fs::create_dir_all(&newest).unwrap();
        write_bytes(
            &newest.join("config.json"),
            br#"{"model_type":"newest","n_positions":4096}"#,
        );
        write_bytes(&model.join("refs/main"), b"missing");

        let metadata = scan_models(&root.0).unwrap()[0].local_metadata();
        assert_eq!(
            metadata
                .selected_revision
                .as_ref()
                .map(|value| value.commit.as_str()),
            Some("newest")
        );
        assert_eq!(metadata.identity.model_type.as_deref(), Some("newest"));
        assert_eq!(metadata.maximum_context_length, Some(4096));
    }

    #[test]
    fn missing_root_is_an_empty_cache() {
        let root = TestDir::new("missing");
        let missing = root.0.join("does-not-exist");
        assert!(scan_models(missing).unwrap().is_empty());
    }

    #[test]
    fn deletes_only_the_selected_model_directory() {
        let root = TestDir::new("delete");
        let selected_path = root.0.join("models--org--selected");
        let other_path = root.0.join("models--org--other");
        fs::create_dir_all(selected_path.join("blobs")).unwrap();
        fs::create_dir_all(other_path.join("blobs")).unwrap();
        write_bytes(&selected_path.join("blobs/selected"), b"selected");
        write_bytes(&other_path.join("blobs/other"), b"other");
        let selected = model_info("org", "selected", selected_path.clone());

        delete_model(&root.0, &selected).unwrap();

        assert!(!selected_path.exists());
        assert!(other_path.join("blobs/other").is_file());
    }

    #[test]
    fn rejects_model_outside_active_cache_root() {
        let root = TestDir::new("delete-root");
        let outside = TestDir::new("delete-outside");
        let outside_path = outside.0.join("models--org--model");
        fs::create_dir_all(&outside_path).unwrap();
        let model = model_info("org", "model", outside_path.clone());

        let error = delete_model(&root.0, &model).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(outside_path.is_dir());
    }

    #[test]
    fn rejects_mismatched_model_identity_and_path() {
        let root = TestDir::new("delete-mismatch");
        let path = root.0.join("models--org--actual");
        fs::create_dir_all(&path).unwrap();
        let wrong_name = model_info("org", "different", path.clone());
        let mut wrong_id = model_info("org", "actual", path.clone());
        wrong_id.id = "other/actual".to_owned();

        assert_eq!(
            delete_model(&root.0, &wrong_name).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            delete_model(&root.0, &wrong_id).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert!(path.is_dir());
    }

    #[test]
    fn missing_model_target_is_an_error() {
        let root = TestDir::new("delete-missing");
        let path = root.0.join("models--org--missing");
        let model = model_info("org", "missing", path);

        let error = delete_model(&root.0, &model).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(
            error
                .to_string()
                .contains("cannot access model cache entry")
        );
    }

    #[test]
    fn rejects_traversal_in_model_path() {
        let root = TestDir::new("delete-traversal");
        let path = root.0.join("unused").join("..").join("models--org--model");
        fs::create_dir_all(root.0.join("models--org--model")).unwrap();
        let model = model_info("org", "model", path);

        let error = delete_model(&root.0, &model).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(root.0.join("models--org--model").is_dir());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symbolic_link_model_entries() {
        use std::os::unix::fs::symlink;

        let root = TestDir::new("delete-symlink-root");
        let outside = TestDir::new("delete-symlink-target");
        let outside_path = outside.0.join("real-model");
        fs::create_dir_all(&outside_path).unwrap();
        write_bytes(&outside_path.join("keep"), b"keep");
        let link = root.0.join("models--org--model");
        symlink(&outside_path, &link).unwrap();
        let model = model_info("org", "model", link.clone());

        let error = delete_model(&root.0, &model).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
        assert!(outside_path.join("keep").is_file());
    }

    fn model_info(organization: &str, name: &str, path: PathBuf) -> ModelInfo {
        ModelInfo {
            id: format!("{organization}/{name}"),
            organization: organization.to_owned(),
            name: name.to_owned(),
            path,
            size_bytes: 0,
            estimated_model_weight_bytes: None,
            snapshot_count: 0,
            revision_count: 0,
            last_modified: None,
        }
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(bytes).unwrap();
    }

    fn write_safetensors(path: &Path, header: &[u8]) {
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[0; 32]).unwrap();
    }
}
