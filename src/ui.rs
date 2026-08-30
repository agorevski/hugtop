use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Margin, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Cell, Clear, List, ListItem, ListState, Paragraph, Row, Table,
        TableState, Wrap,
    },
};

use crate::{
    app::{ActionStatus, App, PendingDelete, VRAM_OVERHEAD_PERCENT},
    cache::ModelInfo,
    gpu::{GpuAllocationEstimate, GpuCountEstimate, GpuDetection, GpuDeviceAllocation},
};

const CYAN: Color = Color::Rgb(66, 211, 255);
const PURPLE: Color = Color::Rgb(183, 120, 255);
const GREEN: Color = Color::Rgb(94, 234, 165);
const MUTED: Color = Color::Rgb(126, 142, 161);
const PANEL: Color = Color::Rgb(32, 40, 54);

pub(crate) fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(Color::Rgb(12, 17, 25))),
        area,
    );

    if area.width < 46 || area.height < 14 {
        draw_compact(frame, area, app);
    } else {
        draw_dashboard(frame, area, app);
    }
    if app.show_help {
        draw_help(frame, area);
    }
    if let Some(pending) = app.pending_delete.as_ref() {
        draw_delete_confirmation(frame, area, pending);
    } else if let Some(deleting) = app.deleting.as_ref() {
        draw_deleting(frame, area, deleting);
    }
}

fn draw_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(if area.height >= 26 { 5 } else { 3 }),
        Constraint::Min(7),
        Constraint::Length(3),
    ])
    .split(area);
    draw_header(frame, rows[0], app);
    draw_summary(frame, rows[1], app);

    let body = if area.width >= 96 {
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).split(rows[2])
    } else {
        Layout::vertical([Constraint::Percentage(57), Constraint::Percentage(43)]).split(rows[2])
    };
    draw_models(frame, body[0], app);
    draw_detail(frame, body[1], app);
    draw_footer(frame, rows[3], app);
}

fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let title = Line::from(vec![
        Span::styled(
            "  ◈ ",
            Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "hugtop",
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" // MODEL CACHE", Style::default().fg(Color::White)),
    ]);
    let filter = if app.editing_filter {
        format!(" FILTER › {}█ ", app.filter_draft)
    } else if app.filter.is_empty() {
        format!(
            " SORT: {} {} ",
            app.sort.label().to_uppercase(),
            app.sort.direction(app.sort_reversed)
        )
    } else {
        format!(
            " FILTER: {}  ·  SORT: {} {} ",
            app.filter,
            app.sort.label(),
            app.sort.direction(app.sort_reversed).to_lowercase()
        )
    };
    let block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(PANEL))
        .title(title)
        .title_bottom(Line::styled(filter, Style::default().fg(GREEN)).alignment(Alignment::Right));
    frame.render_widget(block, area);
}

fn draw_summary(frame: &mut Frame, area: Rect, app: &App) {
    let cards = Layout::horizontal([
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ])
    .split(area);
    let snapshots: usize = app.models.iter().map(|model| model.snapshot_count).sum();
    let revisions: usize = app.models.iter().map(|model| model.revision_count).sum();
    let values = [
        ("MODELS", app.models.len().to_string(), CYAN),
        ("STORAGE", format_bytes(app.total_bytes()), PURPLE),
        ("SNAPSHOTS", snapshots.to_string(), GREEN),
        ("REFERENCES", revisions.to_string(), Color::Yellow),
    ];
    for (area, (label, value, color)) in cards.iter().zip(values) {
        let content = if area.height >= 5 {
            vec![
                Line::styled(label, Style::default().fg(MUTED)),
                Line::styled(
                    value,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ]
        } else {
            vec![Line::from(vec![
                Span::styled(format!("{label} "), Style::default().fg(MUTED)),
                Span::styled(
                    value,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
            ])]
        };
        frame.render_widget(
            Paragraph::new(content)
                .alignment(Alignment::Center)
                .block(panel()),
            *area,
        );
    }
}

fn draw_models(frame: &mut Frame, area: Rect, app: &App) {
    let title = format!(" MODELS  {}/{} ", app.visible.len(), app.models.len());
    if app.visible.is_empty() {
        let message = if app.models.is_empty() {
            "No cached models found\n\nPress r to scan again"
        } else {
            "No models match this filter\n\nPress / to edit the filter"
        };
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .style(Style::default().fg(MUTED))
                .block(panel().title(title))
                .wrap(Wrap { trim: true }),
            area,
        );
        return;
    }

    let rows = app.visible.iter().map(|index| {
        let model = &app.models[*index];
        Row::new(vec![
            Cell::from(model.id.clone()),
            Cell::from(format_bytes(model.size_bytes)),
            Cell::from(
                app.estimated_vram_mib(model)
                    .map(format_mib)
                    .unwrap_or_else(|| "unknown".into()),
            ),
            Cell::from(gpu_count_label(app, model)),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(18),
            Constraint::Length(11),
            Constraint::Length(11),
            Constraint::Length(12),
        ],
    )
    .header(
        Row::new(["REPOSITORY", "CACHE SIZE", "EST. VRAM", "GPUS NEEDED"])
            .style(Style::default().fg(MUTED).add_modifier(Modifier::BOLD))
            .bottom_margin(1),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::Rgb(30, 75, 96))
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )
    .highlight_symbol("▌ ")
    .block(panel().title(title));
    let mut state = TableState::default().with_selected(Some(app.selected));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_detail(frame: &mut Frame, area: Rect, app: &App) {
    let Some(model) = app.selected_model() else {
        frame.render_widget(
            Paragraph::new("Select a model to inspect its cache details")
                .alignment(Alignment::Center)
                .style(Style::default().fg(MUTED))
                .block(panel().title(" INSPECTOR ")),
            area,
        );
        return;
    };

    frame.render_widget(panel().title(" INSPECTOR "), area);
    let inner = area.inner(Margin::new(1, 1));
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let allocation = app.gpu_allocation_estimate(model);
    if inner.height <= 4 {
        frame.render_widget(
            Paragraph::new(vec![
                Line::styled(
                    &model.name,
                    Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
                ),
                Line::styled(
                    compact_vram_summary(allocation.as_ref()),
                    vram_status(app, allocation.as_ref()).1,
                ),
            ]),
            inner,
        );
        return;
    }

    let desired_plan_height = match allocation.as_ref() {
        Some(GpuAllocationEstimate::Allocated { devices, .. }) => u16::try_from(devices.len())
            .unwrap_or(u16::MAX)
            .saturating_add(3)
            .min(9),
        _ => 3,
    };
    let plan_height = desired_plan_height
        .min(inner.height.saturating_sub(2))
        .max(2);
    let rows = Layout::vertical([
        Constraint::Length(inner.height.saturating_sub(plan_height)),
        Constraint::Length(plan_height),
    ])
    .split(inner);

    let mut details = vec![
        Line::styled(
            &model.name,
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
        ),
        Line::from(vec![
            Span::styled("Cache on disk ", Style::default().fg(MUTED)),
            Span::raw(format_bytes(model.size_bytes)),
        ]),
        Line::from(vec![
            Span::styled("Model weights ", Style::default().fg(MUTED)),
            Span::raw(
                model
                    .estimated_model_weight_bytes
                    .map(format_bytes)
                    .unwrap_or_else(|| "unknown (no usable weight artifacts)".into()),
            ),
        ]),
        Line::from(vec![
            Span::styled("Runtime VRAM  ", Style::default().fg(MUTED)),
            Span::raw(vram_requirement_label(app, model)),
        ]),
    ];
    if rows[0].height >= 5 {
        details.push(Line::from(vec![
            Span::styled("Organization  ", Style::default().fg(MUTED)),
            Span::raw(&model.organization),
        ]));
    }
    if rows[0].height >= 6 {
        details.push(Line::from(vec![
            Span::styled("Snapshots     ", Style::default().fg(MUTED)),
            Span::raw(model.snapshot_count.to_string()),
            Span::styled("   Refs  ", Style::default().fg(MUTED)),
            Span::raw(model.revision_count.to_string()),
        ]));
    }
    if rows[0].height >= 7 {
        details.push(Line::from(vec![
            Span::styled("Detected GPUs ", Style::default().fg(MUTED)),
            Span::raw(gpu_inventory_label(&app.gpu_detection)),
        ]));
    }
    if rows[0].height >= 8 {
        details.push(Line::from(vec![
            Span::styled("Location      ", Style::default().fg(MUTED)),
            Span::raw(model.path.display().to_string()),
        ]));
    }
    frame.render_widget(Paragraph::new(details).wrap(Wrap { trim: false }), rows[0]);
    draw_vram_plan(frame, rows[1], app, allocation.as_ref());
}

fn draw_vram_plan(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    allocation: Option<&GpuAllocationEstimate>,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    if area.width < 20 || area.height < 3 {
        let status = vram_status(app, allocation);
        frame.render_widget(
            Paragraph::new(compact_vram_summary(allocation)).style(status.1),
            area,
        );
        return;
    }

    let block = panel().title(" VRAM PLAN ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let (status, status_style) = vram_status(app, allocation);
    frame.render_widget(
        Paragraph::new(status).style(status_style),
        Rect::new(inner.x, inner.y, inner.width, 1),
    );

    let Some(GpuAllocationEstimate::Allocated { devices, .. }) = allocation else {
        return;
    };
    let available_rows = inner.height.saturating_sub(1) as usize;
    if available_rows == 0 {
        return;
    }
    let overflow = devices.len() > available_rows;
    let visible_count = if overflow {
        available_rows.saturating_sub(1)
    } else {
        devices.len().min(available_rows)
    };
    for (row, device) in devices.iter().take(visible_count).enumerate() {
        let free_bytes = selected_gpu_free_bytes(app, device.device_index);
        let style = if free_bytes.is_some_and(|free| free < device.allocated_bytes) {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(GREEN)
        };
        frame.render_widget(
            Paragraph::new(gpu_bar_row(device, inner.width as usize)).style(style),
            Rect::new(
                inner.x,
                inner.y.saturating_add(1 + row as u16),
                inner.width,
                1,
            ),
        );
    }
    if overflow {
        frame.render_widget(
            Paragraph::new(format!(
                "+{} more selected GPUs",
                devices.len().saturating_sub(visible_count)
            ))
            .style(Style::default().fg(MUTED)),
            Rect::new(
                inner.x,
                inner.y.saturating_add(1 + visible_count as u16),
                inner.width,
                1,
            ),
        );
    }
}

fn selected_gpu_free_bytes(app: &App, device_index: usize) -> Option<u64> {
    match &app.gpu_detection {
        GpuDetection::Detected(inventory) => inventory
            .gpus()
            .get(device_index)
            .map(|gpu| gpu.free_mib.saturating_mul(1 << 20)),
        _ => None,
    }
}

fn vram_requirement_label(app: &App, model: &ModelInfo) -> String {
    app.estimated_vram_mib(model)
        .map(|mib| {
            format!(
                "{} (weights + current {VRAM_OVERHEAD_PERCENT}% allowance; activations/KV vary)",
                format_mib(mib)
            )
        })
        .unwrap_or_else(|| "unknown; cache disk size is not a weight estimate".into())
}

fn vram_status(app: &App, allocation: Option<&GpuAllocationEstimate>) -> (String, Style) {
    let success = Style::default().fg(GREEN).add_modifier(Modifier::BOLD);
    let warning = Style::default()
        .fg(Color::Yellow)
        .add_modifier(Modifier::BOLD);
    let failure = Style::default()
        .fg(Color::LightRed)
        .add_modifier(Modifier::BOLD);
    let unknown = Style::default().fg(MUTED).add_modifier(Modifier::BOLD);

    match allocation {
        None => (
            "VRAM unknown: no usable model-weight artifacts".into(),
            unknown,
        ),
        Some(GpuAllocationEstimate::NotRequired) => (
            "No runtime VRAM required: zero-byte weight estimate".into(),
            success,
        ),
        Some(GpuAllocationEstimate::Allocated { devices, .. }) => {
            let low_free: Vec<String> = devices
                .iter()
                .filter_map(|device| {
                    let free = selected_gpu_free_bytes(app, device.device_index)?;
                    (free < device.allocated_bytes).then(|| {
                        format!(
                            "GPU {} ({} free vs {} assigned)",
                            device.device_index,
                            format_bytes(free),
                            format_bytes(device.allocated_bytes)
                        )
                    })
                })
                .collect();
            let mut text = if devices.len() == 1 {
                "Fits 1 GPU · best-case minimum-device set using installed VRAM".into()
            } else {
                format!(
                    "Needs {} GPUs · best-case minimum-device set using installed VRAM",
                    devices.len()
                )
            };
            if !low_free.is_empty() {
                text.push_str(&format!(
                    " · current free memory is below assigned share on {}",
                    low_free.join(", ")
                ));
                (text, warning)
            } else {
                (text, success)
            }
        }
        Some(GpuAllocationEstimate::NoGpu { required_bytes }) => (
            format!(
                "Cannot fit: no NVIDIA GPUs detected · {} required",
                format_bytes(*required_bytes)
            ),
            failure,
        ),
        Some(GpuAllocationEstimate::InsufficientCapacity {
            required_bytes,
            available_bytes,
        }) => (
            format!(
                "Cannot fit in aggregate · {} required · {} installed",
                format_bytes(*required_bytes),
                format_bytes(*available_bytes)
            ),
            failure,
        ),
        Some(GpuAllocationEstimate::Unknown { required_bytes }) => (
            format!(
                "GPU fit unknown: nvidia-smi/detection unavailable · {} required",
                format_bytes(*required_bytes)
            ),
            unknown,
        ),
    }
}

fn compact_vram_summary(allocation: Option<&GpuAllocationEstimate>) -> String {
    match allocation {
        None => "VRAM unknown".into(),
        Some(GpuAllocationEstimate::NotRequired) => "VRAM 0 B · no GPU required".into(),
        Some(GpuAllocationEstimate::Allocated {
            required_bytes,
            devices,
        }) => {
            let share = required_bytes / devices.len().max(1) as u64;
            format!(
                "VRAM {} · {} GPU{} (~{} ea)",
                format_bytes(*required_bytes),
                devices.len(),
                if devices.len() == 1 { "" } else { "s" },
                format_bytes(share)
            )
        }
        Some(GpuAllocationEstimate::NoGpu { required_bytes }) => {
            format!("VRAM {} · no NVIDIA GPUs", format_bytes(*required_bytes))
        }
        Some(GpuAllocationEstimate::InsufficientCapacity { required_bytes, .. }) => format!(
            "VRAM {} · insufficient installed VRAM",
            format_bytes(*required_bytes)
        ),
        Some(GpuAllocationEstimate::Unknown { required_bytes }) => {
            format!(
                "VRAM {} · GPU detection unavailable",
                format_bytes(*required_bytes)
            )
        }
    }
}

fn gpu_bar_row(device: &GpuDeviceAllocation, width: usize) -> String {
    let numbers = allocation_numbers(device.allocated_bytes, device.capacity_bytes);
    let percentage = if device.capacity_bytes == 0 {
        0
    } else {
        ((u128::from(device.allocated_bytes) * 100) / u128::from(device.capacity_bytes)) as u64
    };
    let suffix = format!("{numbers} {percentage}%");
    let base_prefix = format!("GPU {}", device.device_index);
    let mut prefix = base_prefix.clone();
    if width >= 40 {
        let max_name = width
            .saturating_sub(base_prefix.chars().count() + suffix.chars().count() + 14)
            .clamp(6, 24);
        prefix.push(' ');
        prefix.push_str(&truncate_text(&device.name, max_name));
    }
    let fixed = prefix.chars().count() + suffix.chars().count() + 4;
    let bar_width = width.saturating_sub(fixed);
    if bar_width < 3 {
        return format!("{base_prefix} {suffix}");
    }
    let filled = bar_cells(device.allocated_bytes, device.capacity_bytes, bar_width);
    let bar = format!(
        "{}{}",
        "█".repeat(filled),
        "░".repeat(bar_width.saturating_sub(filled))
    );
    format!("{prefix} [{bar}] {suffix}")
}

fn allocation_numbers(allocated: u64, capacity: u64) -> String {
    const GIB: u64 = 1 << 30;
    if allocated.is_multiple_of(GIB) && capacity.is_multiple_of(GIB) {
        format!("{}/{} GiB", allocated / GIB, capacity / GIB)
    } else {
        format!("{}/{}", format_bytes(allocated), format_bytes(capacity))
    }
}

fn bar_cells(allocated: u64, capacity: u64, width: usize) -> usize {
    if width == 0 || allocated == 0 || capacity == 0 {
        return 0;
    }
    if allocated >= capacity {
        return width;
    }
    (((u128::from(allocated) * width as u128) / u128::from(capacity)) as usize)
        .max(1)
        .min(width)
}

fn truncate_text(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.into();
    }
    if max_chars <= 1 {
        return "…".into();
    }
    let mut result: String = value.chars().take(max_chars - 1).collect();
    result.push('…');
    result
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let status = if let Some(status) = &app.action_status {
        Line::styled(
            format!(" {} ", status.message()),
            action_status_style(status),
        )
    } else if let Some(error) = &app.scan_error {
        Line::styled(
            format!(" ⚠ {error} "),
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Line::styled(
            format!(
                " {} · {} visible ",
                app.cache_root.display(),
                format_bytes(app.visible_bytes())
            ),
            Style::default().fg(MUTED),
        )
    };
    let hints = Line::from(vec![
        key("↑↓/jk"),
        Span::raw(" move  "),
        key("/"),
        Span::raw(" filter  "),
        key("s"),
        Span::raw(" sort  "),
        key("S"),
        Span::raw(" reverse  "),
        key("r"),
        Span::raw(" refresh  "),
        Span::styled(
            "d",
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" delete  "),
        key("?"),
        Span::raw(" help  "),
        key("q"),
        Span::raw(" quit "),
    ]);
    frame.render_widget(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(PANEL))
            .title(status)
            .title_bottom(hints.alignment(Alignment::Center)),
        area,
    );
}

fn draw_compact(frame: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(4),
        Constraint::Length(2),
    ])
    .split(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " ◈ hugtop ",
                Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "{} · {}",
                    app.visible.len(),
                    format_bytes(app.visible_bytes())
                ),
                Style::default().fg(MUTED),
            ),
        ])),
        rows[0],
    );
    let items = app.visible.iter().enumerate().map(|(position, index)| {
        let model = &app.models[*index];
        let detail = if position == app.selected {
            let allocation = app.gpu_allocation_estimate(model);
            compact_vram_summary(allocation.as_ref())
        } else {
            format!(
                "VRAM {} · GPUs {}",
                app.estimated_vram_mib(model)
                    .map(format_mib)
                    .unwrap_or_else(|| "unknown".into()),
                gpu_count_label(app, model)
            )
        };
        ListItem::new(Line::from(vec![
            Span::raw(&model.id),
            Span::styled(
                format!("  disk {} · {}", format_bytes(model.size_bytes), detail),
                Style::default().fg(PURPLE),
            ),
        ]))
    });
    let list = List::new(items)
        .block(panel())
        .highlight_symbol("▌ ")
        .highlight_style(Style::default().fg(CYAN).add_modifier(Modifier::BOLD));
    let mut state =
        ListState::default().with_selected((!app.visible.is_empty()).then_some(app.selected));
    frame.render_stateful_widget(list, rows[1], &mut state);
    let prompt = if app.editing_filter {
        format!("/{}█", app.filter_draft)
    } else if let Some(status) = &app.action_status {
        status.message().to_owned()
    } else if let Some(error) = &app.scan_error {
        format!("⚠ {error}")
    } else {
        "↑↓ move  / filter  d delete  ? help  q quit".into()
    };
    let prompt_color = match app.action_status.as_ref() {
        Some(ActionStatus::Success(_)) => GREEN,
        Some(ActionStatus::Failure(_)) => Color::LightRed,
        None if app.scan_error.is_some() => Color::LightRed,
        None => MUTED,
    };
    frame.render_widget(
        Paragraph::new(prompt)
            .style(Style::default().fg(prompt_color))
            .wrap(Wrap { trim: true }),
        rows[2],
    );
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let lines = vec![
        Line::styled(
            "NAVIGATION",
            Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
        ),
        Line::raw("  ↑ / k, ↓ / j       Move selection"),
        Line::raw("  PageUp / PageDown  Move one page"),
        Line::raw("  Home / End         Jump to edge"),
        Line::raw(""),
        Line::styled(
            "ACTIONS",
            Style::default().fg(PURPLE).add_modifier(Modifier::BOLD),
        ),
        Line::raw("  /                  Edit model filter"),
        Line::raw("  s                  Cycle sort order"),
        Line::raw("  S                  Reverse sort direction"),
        Line::raw("  Esc                Clear active filter"),
        Line::raw("  r                  Rescan the cache"),
        Line::raw("  d                  Delete selected model (asks first)"),
        Line::raw("  ?                  Toggle this help"),
        Line::raw("  q / Ctrl-C         Quit"),
        Line::raw(""),
        Line::styled("Press Esc, q, or ? to close", Style::default().fg(GREEN)),
    ];
    let popup = popup_rect(area, 66, (lines.len() as u16).saturating_add(2));
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                panel()
                    .title(" KEYBOARD HELP ")
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(CYAN)),
            )
            .wrap(Wrap { trim: false }),
        popup,
    );
}

fn draw_delete_confirmation(frame: &mut Frame, area: Rect, pending: &PendingDelete) {
    let prompt = vec![
        Line::from(vec![
            Span::styled(
                "Press ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Y",
                Style::default()
                    .fg(Color::Yellow)
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
            ),
            Span::styled(
                " to permanently delete",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::styled(
            "Any other key cancels",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ];

    if area.height < 6 {
        frame.render_widget(Clear, area);
        frame.render_widget(
            Paragraph::new(prompt)
                .style(Style::default().bg(Color::Rgb(12, 17, 25)))
                .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }

    let mut lines = prompt;
    lines.extend([
        Line::raw(""),
        Line::styled(
            "PERMANENT DELETION",
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        ),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(MUTED)),
            Span::styled(
                pending.id.as_str(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Disk space reclaimed: ", Style::default().fg(MUTED)),
            Span::raw(format!(
                "{} ({} bytes)",
                format_bytes(pending.size_bytes),
                pending.size_bytes
            )),
        ]),
        Line::from(vec![
            Span::styled("Full path: ", Style::default().fg(MUTED)),
            Span::raw(pending.path.display().to_string()),
        ]),
        Line::raw(""),
        Line::styled(
            "This is permanent, does not use Trash, and requires re-downloading the model.",
            Style::default()
                .fg(Color::LightRed)
                .add_modifier(Modifier::BOLD),
        ),
    ]);
    let popup = popup_rect(area, 76, (lines.len() as u16).saturating_add(2));
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                panel()
                    .title(" CONFIRM DELETE ")
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(Color::LightRed)),
            )
            .wrap(Wrap { trim: false }),
        popup,
    );
}

fn draw_deleting(frame: &mut Frame, area: Rect, pending: &PendingDelete) {
    let popup = popup_rect(area, 60, 7);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(vec![
            Line::styled(
                "Deleting model cache...",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Line::raw(pending.id.as_str()),
            Line::raw(pending.path.display().to_string()),
        ])
        .alignment(Alignment::Center)
        .block(
            panel()
                .title(" DELETING ")
                .border_type(BorderType::Double)
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .wrap(Wrap { trim: false }),
        popup,
    );
}

fn popup_rect(area: Rect, preferred_width: u16, preferred_height: u16) -> Rect {
    if area.width <= 2 || area.height <= 2 {
        area
    } else {
        centered(
            area,
            preferred_width.min(area.width.saturating_sub(2)),
            preferred_height.min(area.height.saturating_sub(2)),
        )
    }
}

fn action_status_style(status: &ActionStatus) -> Style {
    match status {
        ActionStatus::Success(_) => Style::default().fg(GREEN).add_modifier(Modifier::BOLD),
        ActionStatus::Failure(_) => Style::default()
            .fg(Color::LightRed)
            .add_modifier(Modifier::BOLD),
    }
}

fn panel() -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(PANEL))
}

fn key(value: &'static str) -> Span<'static> {
    Span::styled(
        value,
        Style::default().fg(CYAN).add_modifier(Modifier::BOLD),
    )
}

fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(area.height.saturating_sub(height) / 2),
            Constraint::Length(height),
            Constraint::Min(0),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(area.width.saturating_sub(width) / 2),
            Constraint::Length(width),
            Constraint::Min(0),
        ])
        .split(vertical[1])[1]
}

fn format_mib(mib: u64) -> String {
    format_bytes(mib.saturating_mul(1 << 20))
}

fn gpu_count_label(app: &App, model: &ModelInfo) -> String {
    match app.gpu_count_estimate(model) {
        GpuCountEstimate::NotRequired => "none".into(),
        GpuCountEstimate::Gpus(count) => count.to_string(),
        GpuCountEstimate::InsufficientCapacity {
            available_mib: 0, ..
        } if matches!(app.gpu_detection, GpuDetection::NoGpu) => "no GPUs".into(),
        GpuCountEstimate::InsufficientCapacity { .. } => "insufficient".into(),
        GpuCountEstimate::Unknown => "unknown".into(),
    }
}

fn gpu_inventory_label(detection: &GpuDetection) -> String {
    match detection {
        GpuDetection::Detected(inventory) => {
            let (total, free) = inventory.gpus().iter().fold((0_u64, 0_u64), |sum, gpu| {
                (
                    sum.0.saturating_add(gpu.total_mib),
                    sum.1.saturating_add(gpu.free_mib),
                )
            });
            format!(
                "{} · {} total · {} free",
                inventory.len(),
                format_mib(total),
                format_mib(free)
            )
        }
        GpuDetection::NoGpu => "no NVIDIA GPUs detected".into(),
        GpuDetection::ToolUnavailable { message, .. } => {
            format!("unknown (nvidia-smi unavailable: {message})")
        }
        GpuDetection::CommandFailed { stderr, .. } => {
            format!("unknown (nvidia-smi failed: {stderr})")
        }
        GpuDetection::MalformedOutput(error) => {
            format!("unknown (invalid nvidia-smi output: {})", error.message)
        }
    }
}

pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else if value >= 10.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    const GIB: u64 = 1 << 30;

    fn model(weight_bytes: Option<u64>) -> ModelInfo {
        ModelInfo {
            id: "acme/model".into(),
            organization: "acme".into(),
            name: "model".into(),
            path: "cache/acme/model".into(),
            size_bytes: 80 * GIB,
            estimated_model_weight_bytes: weight_bytes,
            snapshot_count: 1,
            revision_count: 1,
            last_modified: None,
        }
    }

    fn detected(spec: &str) -> GpuDetection {
        GpuDetection::Detected(crate::gpu::parse_nvidia_smi_output(spec).unwrap().unwrap())
    }

    fn app_with(weight_bytes: Option<u64>, detection: GpuDetection) -> App {
        let mut app = App::new("cache".into(), detection);
        app.models = vec![model(weight_bytes)];
        app.visible = vec![0];
        app
    }

    fn buffer_text(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn formats_byte_sizes_readably() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(10 * 1024 * 1024), "10 MiB");
    }

    #[test]
    fn gpu_labels_distinguish_detection_and_capacity_states() {
        assert_eq!(
            gpu_inventory_label(&GpuDetection::NoGpu),
            "no NVIDIA GPUs detected"
        );
        let detected = GpuDetection::Detected(
            crate::gpu::parse_nvidia_smi_output("gpu-a, 8, 4\ngpu-b, 16, 12")
                .unwrap()
                .unwrap(),
        );
        assert_eq!(
            gpu_inventory_label(&detected),
            "2 · 24 MiB total · 16 MiB free"
        );
    }

    #[test]
    fn vram_status_text_covers_every_state() {
        let unknown_weights = app_with(None, GpuDetection::NoGpu);
        assert!(
            vram_status(&unknown_weights, None)
                .0
                .contains("no usable model-weight artifacts")
        );

        let zero = app_with(Some(0), GpuDetection::NoGpu);
        let zero_plan = zero.gpu_allocation_estimate(zero.selected_model().unwrap());
        assert!(
            vram_status(&zero, zero_plan.as_ref())
                .0
                .contains("zero-byte")
        );

        let unavailable = app_with(
            Some(GIB),
            GpuDetection::ToolUnavailable {
                kind: std::io::ErrorKind::NotFound,
                message: "missing".into(),
            },
        );
        let unavailable_plan =
            unavailable.gpu_allocation_estimate(unavailable.selected_model().unwrap());
        assert!(
            vram_status(&unavailable, unavailable_plan.as_ref())
                .0
                .contains("detection unavailable")
        );

        let no_gpu = app_with(Some(GIB), GpuDetection::NoGpu);
        let no_gpu_plan = no_gpu.gpu_allocation_estimate(no_gpu.selected_model().unwrap());
        assert!(
            vram_status(&no_gpu, no_gpu_plan.as_ref())
                .0
                .contains("no NVIDIA GPUs detected")
        );

        let insufficient = app_with(Some(50 * GIB), detected("GPU A, 51200, 51200"));
        let insufficient_plan =
            insufficient.gpu_allocation_estimate(insufficient.selected_model().unwrap());
        assert!(
            vram_status(&insufficient, insufficient_plan.as_ref())
                .0
                .contains("Cannot fit in aggregate")
        );

        let one = app_with(Some(40 * GIB), detected("GPU A, 51200, 51200"));
        let one_plan = one.gpu_allocation_estimate(one.selected_model().unwrap());
        assert!(
            vram_status(&one, one_plan.as_ref())
                .0
                .contains("Fits 1 GPU")
        );

        let multiple = app_with(
            Some(50 * GIB),
            detected("GPU A, 51200, 20000\nGPU B, 51200, 51200"),
        );
        let multiple_plan = multiple.gpu_allocation_estimate(multiple.selected_model().unwrap());
        let status = vram_status(&multiple, multiple_plan.as_ref()).0;
        assert!(status.contains("Needs 2 GPUs"));
        assert!(status.contains("current free memory is below assigned share on GPU 0"));
    }

    #[test]
    fn one_gpu_fit_and_equal_two_gpu_split_use_allocation_api() {
        let one = app_with(Some(40 * GIB), detected("GPU A, 51200, 51200"));
        assert!(matches!(
            one.gpu_allocation_estimate(one.selected_model().unwrap()),
            Some(GpuAllocationEstimate::Allocated { ref devices, .. })
                if devices.len() == 1
        ));

        let two = app_with(
            Some(50 * GIB),
            detected("GPU A, 51200, 51200\nGPU B, 51200, 51200"),
        );
        let Some(GpuAllocationEstimate::Allocated { devices, .. }) =
            two.gpu_allocation_estimate(two.selected_model().unwrap())
        else {
            panic!("expected a two-GPU allocation");
        };
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].allocated_bytes, 30 * GIB);
        assert_eq!(devices[1].allocated_bytes, 30 * GIB);
        assert!(gpu_bar_row(&devices[0], 60).contains("30/50 GiB 60%"));
        assert!(gpu_bar_row(&devices[1], 60).contains("30/50 GiB 60%"));
    }

    #[test]
    fn narrow_gpu_rows_keep_index_bar_and_numbers() {
        let device = GpuDeviceAllocation {
            device_index: 7,
            name: "A very long accelerator name".into(),
            capacity_bytes: 50 * GIB,
            allocated_bytes: 30 * GIB,
        };
        let row = gpu_bar_row(&device, 30);
        assert!(row.contains("GPU 7"));
        assert!(row.contains('█'));
        assert!(row.contains("30/50 GiB 60%"));
        assert!(!row.contains("accelerator"));
    }

    #[test]
    fn bar_cells_handle_exact_fit_and_partial_fill() {
        assert_eq!(bar_cells(50, 50, 10), 10);
        assert_eq!(bar_cells(30, 50, 10), 6);
        assert_eq!(bar_cells(1, 50, 10), 1);
        assert_eq!(bar_cells(0, 50, 10), 0);
        assert_eq!(bar_cells(50, 50, 0), 0);
    }

    #[test]
    fn short_plan_reports_hidden_selected_gpus() {
        let app = app_with(Some(GIB), GpuDetection::NoGpu);
        let devices = (0..4)
            .map(|device_index| GpuDeviceAllocation {
                device_index,
                name: format!("GPU {device_index}"),
                capacity_bytes: 20 * GIB,
                allocated_bytes: 10 * GIB,
            })
            .collect();
        let allocation = GpuAllocationEstimate::Allocated {
            required_bytes: 40 * GIB,
            devices,
        };
        let mut terminal = Terminal::new(TestBackend::new(50, 5)).unwrap();
        terminal
            .draw(|frame| {
                draw_vram_plan(frame, frame.area(), &app, Some(&allocation));
            })
            .unwrap();
        assert!(buffer_text(&terminal).contains("+3 more selected GPUs"));
    }

    #[test]
    fn rendering_smoke_tests_compact_stacked_and_wide_layouts() {
        for (width, height) in [(30, 10), (80, 24), (120, 32), (1, 1)] {
            let app = app_with(
                Some(50 * GIB),
                detected("GPU A, 51200, 51200\nGPU B, 51200, 51200"),
            );
            let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &app)).unwrap();
        }
    }
}
