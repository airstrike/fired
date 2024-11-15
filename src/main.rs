use iced::widget::{center, text};
use iced::{Element, Task};

fn main() -> iced::Result {
    iced::application(Fired::title, Fired::update, Fired::view)
        .antialiasing(true)
        .centered()
        .window(iced::window::Settings {
            min_size: Some((800.0, 600.0).into()),
            size: (800.0, 600.0).into(),
            ..Default::default()
        })
        .run_with(Fired::new)
}

#[derive(Default)]
struct Fired;

#[derive(Debug, Clone)]
pub enum Message {}

impl Fired {
    fn new() -> (Self, Task<Message>) {
        (Self {}, Task::none())
    }
    fn title(&self) -> String {
        "fired • neural network playground".to_string()
    }

    fn update(&mut self, _message: Message) {
        // TODO: Process messages
    }

    fn view(&self) -> Element<'_, Message> {
        center(text("Hello, Fired!").size(50)).into()
    }
}
