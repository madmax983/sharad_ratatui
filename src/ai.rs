use crate::{
    error::{Error, Result, ShadowrunError},
    game_state::GameState,
    message::{self, AIMessage, Message, MessageType, UserCompletionRequest},
    save::get_game_data_dir,
};
use async_openai::{Client, config::OpenAIConfig};
use std::{fs, path::PathBuf, process::Stdio, sync::Arc};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::{Mutex, mpsc},
};
use uuid::Uuid;

#[derive(Debug)]
pub struct GameAI {
    pub client: Client<OpenAIConfig>,
    pub ai_sender: mpsc::UnboundedSender<AIMessage>,
    pub image_sender: mpsc::UnboundedSender<PathBuf>,
    history_lock: Arc<Mutex<()>>,
}

impl Clone for GameAI {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            ai_sender: self.ai_sender.clone(),
            image_sender: self.image_sender.clone(),
            history_lock: Arc::clone(&self.history_lock),
        }
    }
}

impl GameAI {
    pub async fn new(
        api_key: Option<&str>,
        ai_sender: mpsc::UnboundedSender<AIMessage>,
        image_sender: mpsc::UnboundedSender<PathBuf>,
    ) -> Result<Self> {
        let config = match api_key {
            Some(key) => OpenAIConfig::new().with_api_key(key),
            None => OpenAIConfig::new(),
        };

        Ok(Self {
            client: Client::with_config(config),
            ai_sender,
            image_sender,
            history_lock: Arc::new(Mutex::new(())),
        })
    }

    pub async fn start_new_conversation(&self, save_name: &str) -> Result<GameState> {
        let thread_id = Uuid::new_v4().to_string();
        self.save_conversation_history(&thread_id, &[]).await?;

        Ok(GameState::new(
            "codex-exec".to_string(),
            thread_id,
            save_name.to_string(),
        ))
    }

    pub async fn send_message(
        &self,
        mut request: UserCompletionRequest,
        ai_sender: mpsc::UnboundedSender<AIMessage>,
    ) -> Result<()> {
        let mut history = self.fetch_all_messages(&request.state.thread_id).await?;
        let user_message_json = serde_json::to_string(&request.message)?;
        history.push(Message::new(MessageType::User, user_message_json));

        let prompt = self.build_codex_prompt(&request, &history)?;
        let raw_response = self.run_codex_exec(&prompt).await?;
        let game_message = Self::parse_game_message_response(&raw_response)?;

        if let Some(new_character_sheet) = game_message.character_sheet.clone() {
            self.update_character_sheet(&mut request.state, new_character_sheet)?;
            ai_sender
                .send(AIMessage::Save(request.state.clone()))
                .map_err(Error::AISend)?;
        }

        let game_message_json = serde_json::to_string(&game_message)?;
        history.push(Message::new(MessageType::Game, game_message_json));
        self.save_conversation_history(&request.state.thread_id, &history)
            .await?;

        ai_sender
            .send(AIMessage::Response(game_message))
            .map_err(Error::AISend)?;

        Ok(())
    }

    pub fn update_character_sheet(
        &self,
        game_state: &mut GameState,
        new_sheet: crate::character::CharacterSheet,
    ) -> Result<()> {
        game_state.main_character_sheet = Some(new_sheet.clone());

        if let Some(existing_character) = game_state
            .characters
            .iter_mut()
            .find(|c| c.name == new_sheet.name)
        {
            *existing_character = new_sheet;
        } else {
            game_state.characters.push(new_sheet);
        }

        Ok(())
    }

    pub async fn fetch_all_messages(&self, thread_id: &str) -> Result<Vec<Message>> {
        let _guard = self.history_lock.lock().await;
        self.load_conversation_history(thread_id)
    }

    fn build_codex_prompt(
        &self,
        request: &UserCompletionRequest,
        history: &[Message],
    ) -> Result<String> {
        let game_state_json = serde_json::to_string_pretty(&request.state)?;
        let history_json = serde_json::to_string_pretty(history)?;
        let latest_user_json = serde_json::to_string_pretty(&request.message)?;

        Ok(format!(
            "You are the game master for a Shadowrun RPG session.\n\
Respond to the player with ONE JSON object and nothing else.\n\
Do not use markdown code fences.\n\
\n\
Required output shape:\n\
{{\n\
  \"crunch\": \"string\",\n\
  \"fluff\": {{\n\
    \"speakers\": [\n\
      {{\"index\": 0, \"name\": \"Narrator\", \"gender\": \"NonBinary\", \"voice\": null}}\n\
    ],\n\
    \"dialogue\": [\n\
      {{\"speaker_index\": 0, \"text\": \"string\", \"audio\": null}}\n\
    ]\n\
  }},\n\
  \"character_sheet\": null\n\
}}\n\
\n\
Rules:\n\
- Keep the story moving.\n\
- Use the requested language for all narrative text.\n\
- If you include character_sheet, it must be a full valid character sheet object.\n\
- Leave character_sheet as null unless you are creating or fully replacing the sheet.\n\
\n\
Language: {language}\n\
\n\
Current game state JSON:\n\
{game_state_json}\n\
\n\
Conversation history JSON:\n\
{history_json}\n\
\n\
Latest player message JSON:\n\
{latest_user_json}\n",
            language = request.language,
            game_state_json = game_state_json,
            history_json = history_json,
            latest_user_json = latest_user_json,
        ))
    }

    async fn run_codex_exec(&self, prompt: &str) -> Result<String> {
        let output_path =
            std::env::temp_dir().join(format!("sharad-codex-output-{}.txt", Uuid::new_v4()));

        let mut child = Command::new("codex.cmd")
            .arg("exec")
            .arg("--skip-git-repo-check")
            .arg("--output-last-message")
            .arg(&output_path)
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ShadowrunError::AI(format!("Failed to start codex exec: {e}")))?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(prompt.as_bytes())
                .await
                .map_err(|e| ShadowrunError::AI(format!("Failed to write codex prompt: {e}")))?;
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| ShadowrunError::AI(format!("Failed to wait for codex exec: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let message = format!(
                "codex exec failed (status: {}). stderr: {} stdout: {}",
                output.status,
                stderr.trim(),
                stdout.trim()
            );
            return Err(ShadowrunError::AI(message).into());
        }

        let response = tokio::fs::read_to_string(&output_path)
            .await
            .map_err(|e| ShadowrunError::AI(format!("Failed to read codex output: {e}")))?;

        let _ = tokio::fs::remove_file(&output_path).await;

        Ok(response)
    }

    fn parse_game_message_response(raw: &str) -> Result<message::GameMessage> {
        if let Ok(message) = serde_json::from_str::<message::GameMessage>(raw.trim()) {
            return Ok(message);
        }

        if let Some(candidate) = Self::extract_json_candidate(raw) {
            if let Ok(message) = serde_json::from_str::<message::GameMessage>(candidate) {
                return Ok(message);
            }
        }

        Err(ShadowrunError::Game(format!(
            "Failed to parse GameMessage JSON from codex response: {}",
            raw.trim()
        ))
        .into())
    }

    fn extract_json_candidate(raw: &str) -> Option<&str> {
        let start = raw.find('{')?;
        let end = raw.rfind('}')?;
        if end <= start {
            return None;
        }
        Some(&raw[start..=end])
    }

    async fn save_conversation_history(&self, thread_id: &str, messages: &[Message]) -> Result<()> {
        let _guard = self.history_lock.lock().await;
        let path = Self::conversation_file_path(thread_id);

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let serialized = serde_json::to_string_pretty(messages)?;
        fs::write(path, serialized)?;
        Ok(())
    }

    fn load_conversation_history(&self, thread_id: &str) -> Result<Vec<Message>> {
        let path = Self::conversation_file_path(thread_id);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let data = fs::read_to_string(path)?;
        if data.trim().is_empty() {
            return Ok(Vec::new());
        }

        Ok(serde_json::from_str(&data)?)
    }

    fn conversation_file_path(thread_id: &str) -> PathBuf {
        get_game_data_dir()
            .join("conversations")
            .join(format!("{}.json", thread_id))
    }
}

#[cfg(test)]
mod tests {
    use super::GameAI;

    #[test]
    fn parse_game_message_accepts_plain_json() {
        let raw = r#"{
  "crunch": "ok",
  "fluff": {
    "speakers": [
      {"index": 0, "name": "Narrator", "gender": "NonBinary", "voice": null}
    ],
    "dialogue": [
      {"speaker_index": 0, "text": "hello", "audio": null}
    ]
  },
  "character_sheet": null
}"#;

        let parsed = GameAI::parse_game_message_response(raw).expect("expected valid game message");
        assert_eq!(parsed.crunch, "ok");
        assert_eq!(parsed.fluff.dialogue.len(), 1);
    }

    #[test]
    fn parse_game_message_accepts_markdown_wrapped_json() {
        let raw = r#"```json
{
  "crunch": "wrapped",
  "fluff": {
    "speakers": [
      {"index": 0, "name": "Narrator", "gender": "NonBinary", "voice": null}
    ],
    "dialogue": [
      {"speaker_index": 0, "text": "hello", "audio": null}
    ]
  },
  "character_sheet": null
}
```"#;

        let parsed =
            GameAI::parse_game_message_response(raw).expect("expected valid wrapped message");
        assert_eq!(parsed.crunch, "wrapped");
        assert_eq!(parsed.fluff.speakers.len(), 1);
    }
}
