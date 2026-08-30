mod app;
mod cache;
mod gpu;
mod hub;
mod ui;

use std::{
    io::{self, Stdout},
    panic,
    path::PathBuf,
};

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::{
    cursor,
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

pub(crate) type Tui = Terminal<CrosstermBackend<Stdout>>;

#[derive(Debug, Parser)]
#[command(
    name = "hugtop",
    version,
    about = "Explore locally cached Hugging Face models"
)]
struct Cli {
    /// Hugging Face cache directory to inspect instead of the default cache.
    #[arg(long, value_name = "PATH")]
    cache_dir: Option<PathBuf>,

    /// Enrich local cache entries from the Hugging Face Hub.
    #[arg(long)]
    online: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    install_panic_hook();

    let mut terminal = init_terminal().context("failed to initialize terminal")?;
    let app_result = app::run(&mut terminal, cli.cache_dir, cli.online);
    let restore_result = restore_terminal(&mut terminal);

    match (app_result, restore_result) {
        (Err(app_error), Err(restore_error)) => Err(app_error.context(format!(
            "terminal restoration also failed: {restore_error:#}"
        ))),
        (Err(app_error), Ok(())) => Err(app_error),
        (Ok(()), Err(restore_error)) => Err(restore_error).context("failed to restore terminal"),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn init_terminal() -> Result<Tui> {
    enable_raw_mode().context("could not enable raw mode")?;

    let mut stdout = io::stdout();
    if let Err(error) = execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        cursor::Hide
    ) {
        let _ = disable_raw_mode();
        return Err(error).context("could not enter alternate screen");
    }

    match Terminal::new(CrosstermBackend::new(stdout)) {
        Ok(mut terminal) => {
            if let Err(error) = terminal.clear() {
                restore_terminal_best_effort();
                Err(error).context("could not clear terminal")
            } else {
                Ok(terminal)
            }
        }
        Err(error) => {
            restore_terminal_best_effort();
            Err(error).context("could not create terminal backend")
        }
    }
}

fn restore_terminal(terminal: &mut Tui) -> Result<()> {
    let raw_mode_error = disable_raw_mode().err();
    let alternate_screen_error = execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        cursor::Show
    )
    .err();
    let cursor_error = terminal.show_cursor().err();

    if let Some(error) = raw_mode_error {
        Err(error).context("could not disable raw mode")
    } else if let Some(error) = alternate_screen_error {
        Err(error).context("could not leave alternate screen")
    } else if let Some(error) = cursor_error {
        Err(error).context("could not show cursor")
    } else {
        Ok(())
    }
}

fn restore_terminal_best_effort() {
    let _ = disable_raw_mode();
    let _ = execute!(
        io::stdout(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        cursor::Show
    );
}

fn install_panic_hook() {
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        restore_terminal_best_effort();
        previous_hook(panic_info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_is_explicitly_opt_in() {
        assert!(!Cli::try_parse_from(["hugtop"]).unwrap().online);
        assert!(Cli::try_parse_from(["hugtop", "--online"]).unwrap().online);
    }
}
