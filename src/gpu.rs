//! Local NVIDIA GPU discovery and model VRAM fit estimation.
//!
//! Discovery uses `nvidia-smi` and does not require a driver library. VRAM
//! requirements passed to the estimation helpers are assumed to already
//! include framework/runtime overhead.

use std::{
    io,
    num::NonZeroUsize,
    process::{Command, Output},
};

/// One NVIDIA GPU reported by `nvidia-smi`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
    pub compute_capability: Option<ComputeCapability>,
}

/// NVIDIA CUDA compute capability, when reported by `nvidia-smi`.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ComputeCapability {
    pub major: u16,
    pub minor: u16,
}

impl ComputeCapability {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

/// A non-empty collection of detected GPUs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuInventory {
    gpus: Vec<Gpu>,
}

impl GpuInventory {
    pub fn gpus(&self) -> &[Gpu] {
        &self.gpus
    }

    pub fn len(&self) -> usize {
        self.gpus.len()
    }

    /// Estimate the smallest number of the largest GPUs whose combined
    /// capacity satisfies `required_mib`.
    ///
    /// This assumes model weights can be distributed between devices.
    /// `required_mib` should include any desired runtime overhead.
    pub fn gpus_needed(&self, required_mib: u64) -> GpuCountEstimate {
        if required_mib == 0 {
            return GpuCountEstimate::NotRequired;
        }

        let capacities: Vec<u64> = self.gpus.iter().map(|gpu| gpu.total_mib).collect();
        let available_mib = capacities
            .iter()
            .fold(0_u64, |total, capacity| total.saturating_add(*capacity));
        if available_mib < required_mib {
            return GpuCountEstimate::InsufficientCapacity {
                required_mib,
                available_mib,
            };
        }

        let count = minimum_device_count(&capacities, required_mib);
        GpuCountEstimate::Gpus(
            NonZeroUsize::new(count).expect("a positive requirement that fits uses at least one GPU"),
        )
    }

    /// Allocate an already-overhead-adjusted byte requirement across GPUs.
    ///
    /// The minimum possible number of devices is always selected. Among
    /// equally sized minimum sets, the set with the least aggregate unused
    /// capacity is chosen; ties use detection order. Memory is then spread as
    /// evenly as possible, filling a smaller device only when an equal share
    /// would exceed its capacity. Integer-byte remainders use detection order.
    ///
    /// This method does not apply runtime overhead. Callers that need an
    /// allowance must apply it exactly once before calling this method.
    pub fn allocate(&self, required_bytes: u64) -> GpuAllocationEstimate {
        if required_bytes == 0 {
            return GpuAllocationEstimate::NotRequired;
        }

        let capacities: Vec<u64> = self
            .gpus
            .iter()
            .map(|gpu| gpu.total_mib.saturating_mul(BYTES_PER_MIB))
            .collect();
        let available_bytes = capacities
            .iter()
            .fold(0_u64, |total, capacity| total.saturating_add(*capacity));
        if available_bytes < required_bytes {
            return GpuAllocationEstimate::InsufficientCapacity {
                required_bytes,
                available_bytes,
            };
        }

        let minimum_count = minimum_device_count(&capacities, required_bytes);
        let selected = best_device_set(&capacities, required_bytes, minimum_count);
        let allocated = balanced_allocations(&selected, &capacities, required_bytes);
        let devices = selected
            .into_iter()
            .zip(allocated)
            .map(|(device_index, allocated_bytes)| GpuDeviceAllocation {
                device_index,
                name: self.gpus[device_index].name.clone(),
                capacity_bytes: capacities[device_index],
                allocated_bytes,
            })
            .collect();

        GpuAllocationEstimate::Allocated {
            required_bytes,
            devices,
        }
    }
}

/// Result of trying to discover local NVIDIA GPUs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuDetection {
    Detected(GpuInventory),
    NoGpu,
    ToolUnavailable {
        kind: io::ErrorKind,
        message: String,
    },
    CommandFailed {
        exit_code: Option<i32>,
        stderr: String,
    },
    MalformedOutput(GpuParseError),
}

impl GpuDetection {
    /// Estimate GPU count, preserving uncertainty when inventory discovery
    /// could not produce reliable device information.
    pub fn gpus_needed(&self, required_mib: u64) -> GpuCountEstimate {
        if required_mib == 0 {
            return GpuCountEstimate::NotRequired;
        }

        match self {
            Self::Detected(inventory) => inventory.gpus_needed(required_mib),
            Self::NoGpu => GpuCountEstimate::InsufficientCapacity {
                required_mib,
                available_mib: 0,
            },
            Self::ToolUnavailable { .. }
            | Self::CommandFailed { .. }
            | Self::MalformedOutput(_) => GpuCountEstimate::Unknown,
        }
    }

    /// Allocate an already-overhead-adjusted byte requirement while
    /// preserving the distinction between no GPUs and unknown detection.
    pub fn allocate(&self, required_bytes: u64) -> GpuAllocationEstimate {
        if required_bytes == 0 {
            return GpuAllocationEstimate::NotRequired;
        }

        match self {
            Self::Detected(inventory) => inventory.allocate(required_bytes),
            Self::NoGpu => GpuAllocationEstimate::NoGpu { required_bytes },
            Self::ToolUnavailable { .. }
            | Self::CommandFailed { .. }
            | Self::MalformedOutput(_) => GpuAllocationEstimate::Unknown { required_bytes },
        }
    }
}

/// Outcome of a model VRAM fit estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuCountEstimate {
    NotRequired,
    Gpus(NonZeroUsize),
    InsufficientCapacity {
        required_mib: u64,
        available_mib: u64,
    },
    Unknown,
}

/// Per-device share of an allocated model VRAM requirement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDeviceAllocation {
    /// Stable identity within this detected inventory and its original order.
    pub device_index: usize,
    pub name: String,
    /// Installed VRAM capacity.
    pub capacity_bytes: u64,
    pub allocated_bytes: u64,
}

/// Outcome of assigning a model VRAM requirement to detected GPUs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuAllocationEstimate {
    NotRequired,
    Allocated {
        required_bytes: u64,
        devices: Vec<GpuDeviceAllocation>,
    },
    NoGpu {
        required_bytes: u64,
    },
    InsufficientCapacity {
        required_bytes: u64,
        available_bytes: u64,
    },
    Unknown {
        required_bytes: u64,
    },
}

/// Details about an invalid successful `nvidia-smi` response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuParseError {
    pub line: usize,
    pub message: String,
}

/// Detect NVIDIA GPUs installed on the local machine.
pub fn detect_nvidia_gpus() -> GpuDetection {
    let extended = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();
    if extended
        .as_ref()
        .is_ok_and(|output| output.status.success())
    {
        return classify_command_result(extended);
    }

    classify_command_result(
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ])
            .output(),
    )
}

/// Parse output from the query used by [`detect_nvidia_gpus`].
///
/// `Ok(None)` means the command reported no devices. Non-empty invalid output
/// is returned as an error rather than silently producing an empty inventory.
pub fn parse_nvidia_smi_output(output: &str) -> Result<Option<GpuInventory>, GpuParseError> {
    let output = output.trim();
    if output.is_empty() || output.eq_ignore_ascii_case("no devices were found") {
        return Ok(None);
    }

    let mut gpus = Vec::new();
    for (index, line) in output.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line).map_err(|message| GpuParseError {
            line: line_number,
            message,
        })?;
        if fields.len() != 3 && fields.len() != 4 {
            return Err(GpuParseError {
                line: line_number,
                message: format!("expected 3 or 4 CSV fields, found {}", fields.len()),
            });
        }

        let name = fields[0].trim().to_owned();
        if name.is_empty() {
            return Err(GpuParseError {
                line: line_number,
                message: "GPU name is empty".to_owned(),
            });
        }
        let total_mib = parse_mib(&fields[1], line_number, "total")?;
        let free_mib = parse_mib(&fields[2], line_number, "free")?;
        if total_mib == 0 {
            return Err(GpuParseError {
                line: line_number,
                message: "total memory must be greater than zero".to_owned(),
            });
        }
        if free_mib > total_mib {
            return Err(GpuParseError {
                line: line_number,
                message: "free memory exceeds total memory".to_owned(),
            });
        }
        let compute_capability = fields
            .get(3)
            .map(|field| parse_compute_capability(field, line_number))
            .transpose()?
            .flatten();
        gpus.push(Gpu {
            name,
            total_mib,
            free_mib,
            compute_capability,
        });
    }

    if gpus.is_empty() {
        Ok(None)
    } else {
        Ok(Some(GpuInventory { gpus }))
    }
}

const BYTES_PER_MIB: u64 = 1024 * 1024;

/// Smallest number of devices (largest-first) whose combined capacity in the
/// caller's unit reaches `required`. Falls back to the full count when the
/// capacities cannot satisfy `required`.
fn minimum_device_count(capacities: &[u64], required: u64) -> usize {
    let mut descending = capacities.to_vec();
    descending.sort_unstable_by(|left, right| right.cmp(left));
    let mut total = 0_u64;
    for (index, capacity) in descending.into_iter().enumerate() {
        total = total.saturating_add(capacity);
        if total >= required {
            return index + 1;
        }
    }
    capacities.len()
}

fn best_device_set(capacities: &[u64], required_bytes: u64, count: usize) -> Vec<usize> {
    fn visit(
        capacities: &[u64],
        required_bytes: u64,
        count: usize,
        next: usize,
        current: &mut Vec<usize>,
        current_capacity: u128,
        best: &mut Option<(u128, Vec<usize>)>,
    ) {
        if current.len() == count {
            if current_capacity >= u128::from(required_bytes)
                && best.as_ref().is_none_or(|(best_capacity, best_indices)| {
                    current_capacity < *best_capacity
                        || (current_capacity == *best_capacity
                            && current.as_slice() < best_indices.as_slice())
                })
            {
                *best = Some((current_capacity, current.clone()));
            }
            return;
        }

        let still_needed = count - current.len();
        if capacities.len() - next < still_needed {
            return;
        }
        for index in next..=capacities.len() - still_needed {
            current.push(index);
            visit(
                capacities,
                required_bytes,
                count,
                index + 1,
                current,
                current_capacity + u128::from(capacities[index]),
                best,
            );
            current.pop();
        }
    }

    let mut best = None;
    visit(
        capacities,
        required_bytes,
        count,
        0,
        &mut Vec::with_capacity(count),
        0,
        &mut best,
    );
    best.expect("minimum device count must have a fitting set")
        .1
}

fn balanced_allocations(selected: &[usize], capacities: &[u64], required_bytes: u64) -> Vec<u64> {
    let mut allocations = vec![0; selected.len()];
    let mut active: Vec<usize> = (0..selected.len()).collect();
    let mut remaining = required_bytes;

    while !active.is_empty() {
        let equal_share = remaining / active.len() as u64;
        let constrained: Vec<usize> = active
            .iter()
            .copied()
            .filter(|position| capacities[selected[*position]] <= equal_share)
            .collect();

        if constrained.is_empty() {
            let remainder = remaining % active.len() as u64;
            for (order, position) in active.into_iter().enumerate() {
                allocations[position] = equal_share + u64::from((order as u64) < remainder);
            }
            break;
        }

        for position in constrained {
            let capacity = capacities[selected[position]];
            allocations[position] = capacity;
            remaining -= capacity;
            active.retain(|active_position| *active_position != position);
        }
    }

    allocations
}

fn classify_command_result(result: io::Result<Output>) -> GpuDetection {
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            return GpuDetection::ToolUnavailable {
                kind: error.kind(),
                message: error.to_string(),
            };
        }
    };

    if !output.status.success() {
        return GpuDetection::CommandFailed {
            exit_code: output.status.code(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        };
    }

    let stdout = match String::from_utf8(output.stdout) {
        Ok(stdout) => stdout,
        Err(error) => {
            return GpuDetection::MalformedOutput(GpuParseError {
                line: 0,
                message: format!("output is not valid UTF-8: {error}"),
            });
        }
    };
    match parse_nvidia_smi_output(&stdout) {
        Ok(Some(inventory)) => GpuDetection::Detected(inventory),
        Ok(None) => GpuDetection::NoGpu,
        Err(error) => GpuDetection::MalformedOutput(error),
    }
}

fn parse_mib(value: &str, line: usize, label: &str) -> Result<u64, GpuParseError> {
    value.trim().parse().map_err(|_| GpuParseError {
        line,
        message: format!("{label} memory is not a valid MiB value"),
    })
}

fn parse_compute_capability(
    value: &str,
    line: usize,
) -> Result<Option<ComputeCapability>, GpuParseError> {
    let value = value.trim();
    if value.is_empty()
        || value.eq_ignore_ascii_case("n/a")
        || value.eq_ignore_ascii_case("[not supported]")
    {
        return Ok(None);
    }
    let (major, minor) = value.split_once('.').ok_or_else(|| GpuParseError {
        line,
        message: "compute capability is not in major.minor form".to_owned(),
    })?;
    let major = major.parse().map_err(|_| GpuParseError {
        line,
        message: "compute capability major version is invalid".to_owned(),
    })?;
    let minor = minor.parse().map_err(|_| GpuParseError {
        line,
        message: "compute capability minor version is invalid".to_owned(),
    })?;
    Ok(Some(ComputeCapability { major, minor }))
}

fn parse_csv_line(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;

    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(field.trim().to_owned());
                field.clear();
            }
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".to_owned());
    }
    fields.push(field.trim().to_owned());
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn inventory(capacities_gib: &[u64]) -> GpuInventory {
        let output = capacities_gib
            .iter()
            .enumerate()
            .map(|(index, capacity)| {
                format!("gpu-{index}, {}, {}", capacity * 1024, capacity * 1024)
            })
            .collect::<Vec<_>>()
            .join("\n");
        parse_nvidia_smi_output(&output).unwrap().unwrap()
    }

    fn allocated_devices(estimate: GpuAllocationEstimate) -> Vec<GpuDeviceAllocation> {
        match estimate {
            GpuAllocationEstimate::Allocated { devices, .. } => devices,
            other => panic!("expected allocation, got {other:?}"),
        }
    }

    #[test]
    fn parses_heterogeneous_gpus() {
        let inventory = parse_nvidia_smi_output(
            "NVIDIA RTX 4090, 24564, 22000\n\"NVIDIA A100, PCIe\", 40960, 1024\n",
        )
        .unwrap()
        .unwrap();

        assert_eq!(inventory.len(), 2);
        assert_eq!(
            inventory.gpus()[0],
            Gpu {
                name: "NVIDIA RTX 4090".to_owned(),
                total_mib: 24564,
                free_mib: 22000,
                compute_capability: None,
            }
        );
        assert_eq!(inventory.gpus()[1].name, "NVIDIA A100, PCIe");
        assert_eq!(inventory.gpus()[1].total_mib, 40960);
    }

    #[test]
    fn parses_optional_compute_capability_and_old_output() {
        let inventory = parse_nvidia_smi_output("new, 100, 90, 8.9\nold, 200, 100, N/A")
            .unwrap()
            .unwrap();
        assert_eq!(
            inventory.gpus()[0].compute_capability,
            Some(ComputeCapability::new(8, 9))
        );
        assert_eq!(inventory.gpus()[1].compute_capability, None);
        assert_eq!(
            parse_nvidia_smi_output("legacy, 100, 90")
                .unwrap()
                .unwrap()
                .gpus()[0]
                .compute_capability,
            None
        );
    }

    #[test]
    fn uses_minimum_number_of_largest_gpus() {
        let inventory =
            parse_nvidia_smi_output("small, 8000, 7000\nlarge, 24000, 4000\nmid, 16000, 12000")
                .unwrap()
                .unwrap();

        assert_eq!(
            inventory.gpus_needed(30_000),
            GpuCountEstimate::Gpus(NonZeroUsize::new(2).unwrap())
        );
        assert_eq!(
            inventory.gpus_needed(15_000),
            GpuCountEstimate::Gpus(NonZeroUsize::new(1).unwrap())
        );
    }

    #[test]
    fn reports_insufficient_capacity() {
        let inventory = parse_nvidia_smi_output("gpu-a, 10000, 8000\ngpu-b, 12000, 6000")
            .unwrap()
            .unwrap();

        assert_eq!(
            inventory.gpus_needed(25_000),
            GpuCountEstimate::InsufficientCapacity {
                required_mib: 25_000,
                available_mib: 22_000,
            }
        );
    }

    #[test]
    fn zero_requirement_needs_no_gpu_even_when_detection_is_unknown() {
        let unavailable = GpuDetection::ToolUnavailable {
            kind: io::ErrorKind::NotFound,
            message: "missing".to_owned(),
        };
        assert_eq!(unavailable.gpus_needed(0), GpuCountEstimate::NotRequired);
    }

    #[test]
    fn distinguishes_no_gpu_and_malformed_output() {
        assert_eq!(parse_nvidia_smi_output(" \n").unwrap(), None);
        assert_eq!(
            parse_nvidia_smi_output("No devices were found").unwrap(),
            None
        );
        let malformed = parse_nvidia_smi_output("gpu, not-a-number, 100").unwrap_err();
        assert_eq!(malformed.line, 1);
        assert!(malformed.message.contains("total memory"));
    }

    #[test]
    fn validates_memory_relationships_and_csv() {
        assert!(parse_nvidia_smi_output("gpu, 100, 101").is_err());
        assert!(parse_nvidia_smi_output("\"gpu, 100, 50").is_err());
        assert!(parse_nvidia_smi_output("gpu, 100").is_err());
    }

    #[test]
    fn allocation_fits_one_gpu_and_uses_best_fit() {
        let devices = allocated_devices(inventory(&[80, 50, 100]).allocate(45 * GIB));

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_index, 1);
        assert_eq!(devices[0].name, "gpu-1");
        assert_eq!(devices[0].capacity_bytes, 50 * GIB);
        assert_eq!(devices[0].allocated_bytes, 45 * GIB);
    }

    #[test]
    fn allocation_balances_sixty_gib_across_two_fifty_gib_gpus() {
        let devices = allocated_devices(inventory(&[50, 50]).allocate(60 * GIB));

        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].allocated_bytes, 30 * GIB);
        assert_eq!(devices[1].allocated_bytes, 30 * GIB);
    }

    #[test]
    fn allocation_accepts_exact_capacity_boundary() {
        let devices = allocated_devices(inventory(&[50, 30]).allocate(50 * GIB));

        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].device_index, 0);
        assert_eq!(devices[0].allocated_bytes, devices[0].capacity_bytes);
    }

    #[test]
    fn allocation_balances_101_gib_across_three_fifty_gib_gpus() {
        let devices = allocated_devices(inventory(&[50, 50, 50]).allocate(101 * GIB));

        assert_eq!(devices.len(), 3);
        assert_eq!(
            devices
                .iter()
                .map(|device| device.allocated_bytes)
                .sum::<u64>(),
            101 * GIB
        );
        let minimum = devices
            .iter()
            .map(|device| device.allocated_bytes)
            .min()
            .unwrap();
        let maximum = devices
            .iter()
            .map(|device| device.allocated_bytes)
            .max()
            .unwrap();
        assert!(maximum - minimum <= 1);
    }

    #[test]
    fn allocation_respects_heterogeneous_capacity_constraints() {
        let devices = allocated_devices(inventory(&[20, 50]).allocate(60 * GIB));

        assert_eq!(devices[0].allocated_bytes, 20 * GIB);
        assert_eq!(devices[1].allocated_bytes, 40 * GIB);
        assert!(
            devices
                .iter()
                .all(|device| device.allocated_bytes <= device.capacity_bytes)
        );
    }

    #[test]
    fn allocation_uses_no_more_gpus_than_needed() {
        let devices = allocated_devices(inventory(&[50, 50, 50, 50]).allocate(75 * GIB));

        assert_eq!(devices.len(), 2);
        assert_eq!(
            devices
                .iter()
                .map(|device| device.device_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn allocation_reports_insufficient_total_capacity() {
        assert_eq!(
            inventory(&[20, 30]).allocate(51 * GIB),
            GpuAllocationEstimate::InsufficientCapacity {
                required_bytes: 51 * GIB,
                available_bytes: 50 * GIB,
            }
        );
    }

    #[test]
    fn allocation_distinguishes_zero_no_gpu_and_unknown() {
        let unknown = GpuDetection::ToolUnavailable {
            kind: io::ErrorKind::NotFound,
            message: "missing".to_owned(),
        };

        assert_eq!(unknown.allocate(0), GpuAllocationEstimate::NotRequired);
        assert_eq!(
            GpuDetection::NoGpu.allocate(GIB),
            GpuAllocationEstimate::NoGpu {
                required_bytes: GIB
            }
        );
        assert_eq!(
            unknown.allocate(GIB),
            GpuAllocationEstimate::Unknown {
                required_bytes: GIB
            }
        );
    }

    #[test]
    fn allocation_selection_is_stable_for_equal_sets() {
        let devices = allocated_devices(inventory(&[50, 50, 50]).allocate(60 * GIB));

        assert_eq!(
            devices
                .iter()
                .map(|device| device.device_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn allocation_chooses_smallest_fitting_minimum_set() {
        let devices = allocated_devices(inventory(&[60, 40, 40]).allocate(80 * GIB));

        assert_eq!(
            devices
                .iter()
                .map(|device| device.device_index)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn unavailable_command_is_distinct_and_estimate_is_unknown() {
        let detection = classify_command_result(Err(io::Error::new(
            io::ErrorKind::NotFound,
            "nvidia-smi missing",
        )));
        assert!(matches!(
            detection,
            GpuDetection::ToolUnavailable {
                kind: io::ErrorKind::NotFound,
                ..
            }
        ));
        assert_eq!(detection.gpus_needed(1), GpuCountEstimate::Unknown);
    }

    #[cfg(unix)]
    #[test]
    fn command_failure_is_distinct() {
        use std::os::unix::process::ExitStatusExt;

        let detection = classify_command_result(Ok(Output {
            status: std::process::ExitStatus::from_raw(2 << 8),
            stdout: Vec::new(),
            stderr: b"driver unavailable\n".to_vec(),
        }));
        assert_eq!(
            detection,
            GpuDetection::CommandFailed {
                exit_code: Some(2),
                stderr: "driver unavailable".to_owned(),
            }
        );
    }
}
