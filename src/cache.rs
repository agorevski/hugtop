//! Discovery and inspection of the on-disk Hugging Face Hub cache.
//!
//! This module deliberately uses only the standard library. Cache entries are
//! treated as untrusted filesystem data: malformed names and unreadable
//! children are skipped rather than making the whole scan fail.

use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{OsStr, OsString},
    fs, io,
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

const MODEL_PREFIX: &str = "models--";

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WeightFormat {
    SafeTensors,
    Pytorch,
    Gguf,
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
            && let Ok(contents) = fs::read_to_string(path)
        {
            let revision = contents.trim();
            if !revision.is_empty() {
                let candidate = snapshots_root.join(revision);
                if candidate
                    .file_name()
                    .is_some_and(|name| name == OsStr::new(revision))
                    && candidate.is_dir()
                {
                    referenced.push(candidate);
                }
            }
        }
    }
    referenced
}

fn newest_path(paths: Vec<PathBuf>) -> Option<PathBuf> {
    paths.into_iter().max_by_key(|path| {
        collect_tree_stats(path)
            .last_modified
            .unwrap_or(SystemTime::UNIX_EPOCH)
    })
}

fn weight_artifact(file_name: &str) -> Option<WeightArtifact> {
    let lower = file_name.to_ascii_lowercase();
    if is_non_model_artifact(&lower) {
        return None;
    }
    let (stem, format) = if let Some(stem) = lower.strip_suffix(".safetensors") {
        (stem, WeightFormat::SafeTensors)
    } else if let Some(stem) = lower.strip_suffix(".gguf") {
        (stem, WeightFormat::Gguf)
    } else if let Some(stem) = lower.strip_suffix(".bin") {
        (stem, WeightFormat::Pytorch)
    } else if let Some(stem) = lower.strip_suffix(".pth") {
        (stem, WeightFormat::Pytorch)
    } else {
        (lower.strip_suffix(".pt")?, WeightFormat::Pytorch)
    };
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
}
