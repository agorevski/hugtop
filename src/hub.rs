use std::fmt;
use std::time::Duration;

use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde::Deserialize;

const HUB_API_BASE: &str = "https://huggingface.co/api/models";
const MAX_RESPONSE_BYTES: u64 = 2 * 1024 * 1024;
const USER_AGENT: &str = concat!("hugtop/", env!("CARGO_PKG_VERSION"));

#[derive(Clone)]
pub struct HubClient {
    agent: ureq::Agent,
}

impl HubClient {
    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .https_only(true)
            .user_agent(USER_AGENT)
            .accept("application/json")
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_recv_response(Some(Duration::from_secs(10)))
            .timeout_recv_body(Some(Duration::from_secs(10)))
            .timeout_global(Some(Duration::from_secs(20)))
            .build();

        Self {
            agent: ureq::Agent::new_with_config(config),
        }
    }

    /// Fetches public metadata without sending credentials.
    pub fn fetch_model(&self, repo_id: &str) -> Result<ModelMetadata, HubError> {
        self.fetch_model_with_token(repo_id, None)
    }

    /// Fetches metadata with an optional explicitly supplied Hugging Face token.
    ///
    /// The token is only placed in the Authorization header and is never retained
    /// by this client or included in its errors.
    pub fn fetch_model_with_token(
        &self,
        repo_id: &str,
        token: Option<&str>,
    ) -> Result<ModelMetadata, HubError> {
        let url = model_api_url(repo_id)?;
        let mut request = self.agent.get(&url);
        if let Some(token) = token.filter(|token| !token.is_empty()) {
            request = request.header("Authorization", format!("Bearer {token}"));
        }

        let mut response = request.call().map_err(map_http_error)?;
        let body = response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES + 1)
            .read_to_string()
            .map_err(map_http_error)?;

        if body.len() as u64 > MAX_RESPONSE_BYTES {
            return Err(HubError::ResponseTooLarge {
                limit_bytes: MAX_RESPONSE_BYTES,
            });
        }

        parse_model_response(&body)
    }
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub repo_id: String,
    pub latest_revision: Option<String>,
    pub last_modified: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library: Option<String>,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub gated: GatedStatus,
    pub private: bool,
    pub disabled: bool,
    pub deprecated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum GatedStatus {
    #[default]
    No,
    Yes,
    Manual,
    Other(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionStatus {
    UpToDate,
    Outdated,
    Unknown,
}

pub fn compare_revision(
    local_revision: Option<&str>,
    latest_revision: Option<&str>,
) -> RevisionStatus {
    let (Some(local), Some(latest)) = (
        normalized_revision(local_revision),
        normalized_revision(latest_revision),
    ) else {
        return RevisionStatus::Unknown;
    };

    if local.eq_ignore_ascii_case(latest)
        || (is_commit_id(local)
            && is_commit_id(latest)
            && latest.len() >= local.len()
            && latest[..local.len()].eq_ignore_ascii_case(local))
    {
        RevisionStatus::UpToDate
    } else if is_commit_id(local) && is_commit_id(latest) {
        RevisionStatus::Outdated
    } else {
        RevisionStatus::Unknown
    }
}

pub fn parse_model_response(json: &str) -> Result<ModelMetadata, HubError> {
    let raw: ApiModel = serde_json::from_str(json).map_err(|error| HubError::InvalidResponse {
        message: error.to_string(),
    })?;

    let repo_id = raw
        .model_id
        .or(raw.id)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| HubError::InvalidResponse {
            message: "response did not contain a model id".to_string(),
        })?;
    let license = raw
        .license
        .or_else(|| raw.card_data.as_ref().and_then(|card| card.license.clone()))
        .or_else(|| {
            raw.tags
                .iter()
                .find_map(|tag| tag.strip_prefix("license:").map(str::to_owned))
        });
    let deprecated = raw.deprecated || raw.card_data.as_ref().is_some_and(|card| card.deprecated);

    Ok(ModelMetadata {
        repo_id,
        latest_revision: raw.sha,
        last_modified: raw.last_modified,
        pipeline_tag: raw.pipeline_tag,
        library: raw.library_name,
        tags: raw.tags,
        license,
        gated: raw.gated.into(),
        private: raw.private,
        disabled: raw.disabled,
        deprecated,
    })
}

fn model_api_url(repo_id: &str) -> Result<String, HubError> {
    let repo_id = repo_id.trim();
    let segments: Vec<_> = repo_id.split('/').collect();
    if repo_id.is_empty()
        || repo_id.len() > 192
        || segments.len() > 2
        || segments.iter().any(|segment| {
            segment.is_empty()
                || *segment == "."
                || *segment == ".."
                || segment.chars().any(char::is_control)
        })
    {
        return Err(HubError::InvalidRepoId);
    }

    let encoded = segments
        .into_iter()
        .map(|segment| utf8_percent_encode(segment, NON_ALPHANUMERIC).to_string())
        .collect::<Vec<_>>()
        .join("/");
    Ok(format!("{HUB_API_BASE}/{encoded}"))
}

fn normalized_revision(revision: Option<&str>) -> Option<&str> {
    revision
        .map(str::trim)
        .filter(|revision| !revision.is_empty())
}

fn is_commit_id(revision: &str) -> bool {
    revision.len() >= 7 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn map_http_error(error: ureq::Error) -> HubError {
    match error {
        ureq::Error::StatusCode(401) => HubError::AuthenticationRequired,
        ureq::Error::StatusCode(403) => HubError::AccessDenied,
        ureq::Error::StatusCode(404) => HubError::NotFound,
        ureq::Error::StatusCode(429) => HubError::RateLimited,
        ureq::Error::StatusCode(status) => HubError::HttpStatus { status },
        ureq::Error::Timeout(_) => HubError::Timeout,
        ureq::Error::BodyExceedsLimit(_) => HubError::ResponseTooLarge {
            limit_bytes: MAX_RESPONSE_BYTES,
        },
        other => HubError::Network {
            message: other.to_string(),
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    InvalidRepoId,
    AuthenticationRequired,
    AccessDenied,
    NotFound,
    RateLimited,
    HttpStatus { status: u16 },
    Timeout,
    ResponseTooLarge { limit_bytes: u64 },
    Network { message: String },
    InvalidResponse { message: String },
}

impl fmt::Display for HubError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRepoId => write!(formatter, "invalid Hugging Face repository id"),
            Self::AuthenticationRequired => {
                write!(formatter, "Hugging Face authentication is required")
            }
            Self::AccessDenied => write!(formatter, "access to this Hugging Face model was denied"),
            Self::NotFound => write!(formatter, "Hugging Face model was not found"),
            Self::RateLimited => write!(formatter, "Hugging Face API rate limit was reached"),
            Self::HttpStatus { status } => {
                write!(formatter, "Hugging Face API returned HTTP status {status}")
            }
            Self::Timeout => write!(formatter, "Hugging Face API request timed out"),
            Self::ResponseTooLarge { limit_bytes } => write!(
                formatter,
                "Hugging Face API response exceeded the {limit_bytes}-byte limit"
            ),
            Self::Network { message } => {
                write!(
                    formatter,
                    "could not contact the Hugging Face API: {message}"
                )
            }
            Self::InvalidResponse { message } => {
                write!(
                    formatter,
                    "Hugging Face API returned invalid metadata: {message}"
                )
            }
        }
    }
}

impl std::error::Error for HubError {}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiModel {
    id: Option<String>,
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    sha: Option<String>,
    last_modified: Option<String>,
    #[serde(rename = "pipeline_tag", alias = "pipelineTag")]
    pipeline_tag: Option<String>,
    #[serde(rename = "library_name", alias = "libraryName")]
    library_name: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    license: Option<String>,
    #[serde(default)]
    gated: ApiGated,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    deprecated: bool,
    card_data: Option<CardData>,
}

#[derive(Debug, Deserialize)]
struct CardData {
    license: Option<String>,
    #[serde(default)]
    deprecated: bool,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
enum ApiGated {
    Boolean(bool),
    Text(String),
    #[default]
    Missing,
}

impl From<ApiGated> for GatedStatus {
    fn from(value: ApiGated) -> Self {
        match value {
            ApiGated::Boolean(false) | ApiGated::Missing => Self::No,
            ApiGated::Boolean(true) => Self::Yes,
            ApiGated::Text(value) if value.eq_ignore_ascii_case("manual") => Self::Manual,
            ApiGated::Text(value) if value.eq_ignore_ascii_case("false") => Self::No,
            ApiGated::Text(value) if value.eq_ignore_ascii_case("true") => Self::Yes,
            ApiGated::Text(value) => Self::Other(value),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_normal_model_metadata() {
        let metadata = parse_model_response(
            r#"{
                "id": "acme/example",
                "sha": "0123456789abcdef0123456789abcdef01234567",
                "lastModified": "2026-08-29T12:34:56.000Z",
                "pipeline_tag": "text-generation",
                "library_name": "transformers",
                "tags": ["transformers", "license:apache-2.0"],
                "private": false,
                "gated": false
            }"#,
        )
        .unwrap();

        assert_eq!(metadata.repo_id, "acme/example");
        assert_eq!(
            metadata.latest_revision.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(metadata.pipeline_tag.as_deref(), Some("text-generation"));
        assert_eq!(metadata.library.as_deref(), Some("transformers"));
        assert_eq!(metadata.license.as_deref(), Some("apache-2.0"));
        assert_eq!(metadata.gated, GatedStatus::No);
        assert!(!metadata.private);
    }

    #[test]
    fn missing_optional_fields_have_safe_defaults() {
        let metadata = parse_model_response(r#"{"modelId":"acme/minimal"}"#).unwrap();

        assert_eq!(metadata.repo_id, "acme/minimal");
        assert_eq!(metadata.latest_revision, None);
        assert!(metadata.tags.is_empty());
        assert_eq!(metadata.gated, GatedStatus::No);
        assert!(!metadata.disabled);
        assert!(!metadata.deprecated);
    }

    #[test]
    fn parses_gated_and_deprecation_state() {
        let metadata = parse_model_response(
            r#"{
                "id":"acme/restricted",
                "gated":"manual",
                "private":true,
                "disabled":true,
                "cardData":{"license":"other","deprecated":true}
            }"#,
        )
        .unwrap();

        assert_eq!(metadata.gated, GatedStatus::Manual);
        assert!(metadata.private);
        assert!(metadata.disabled);
        assert!(metadata.deprecated);
        assert_eq!(metadata.license.as_deref(), Some("other"));
    }

    #[test]
    fn malformed_or_incomplete_json_is_rejected() {
        assert!(matches!(
            parse_model_response("{not-json"),
            Err(HubError::InvalidResponse { .. })
        ));
        assert!(matches!(
            parse_model_response(r#"{"sha":"abc"}"#),
            Err(HubError::InvalidResponse { .. })
        ));
    }

    #[test]
    fn compares_commit_revisions() {
        let latest = Some("0123456789abcdef0123456789abcdef01234567");
        assert_eq!(
            compare_revision(Some("0123456789abcdef0123456789abcdef01234567"), latest),
            RevisionStatus::UpToDate
        );
        assert_eq!(
            compare_revision(Some("0123456"), latest),
            RevisionStatus::UpToDate
        );
        assert_eq!(
            compare_revision(Some("fedcba9876543210"), latest),
            RevisionStatus::Outdated
        );
        assert_eq!(
            compare_revision(Some("main"), latest),
            RevisionStatus::Unknown
        );
        assert_eq!(compare_revision(None, latest), RevisionStatus::Unknown);
    }

    #[test]
    fn builds_safe_model_urls() {
        assert_eq!(
            model_api_url("acme/model name").unwrap(),
            "https://huggingface.co/api/models/acme/model%20name"
        );
        assert_eq!(
            model_api_url("model#fragment").unwrap(),
            "https://huggingface.co/api/models/model%23fragment"
        );
        assert!(matches!(
            model_api_url("../secret"),
            Err(HubError::InvalidRepoId)
        ));
        assert!(matches!(
            model_api_url("a/b/c"),
            Err(HubError::InvalidRepoId)
        ));
    }

    #[test]
    fn maps_status_and_oversize_errors_for_ui() {
        assert_eq!(
            map_http_error(ureq::Error::StatusCode(404)),
            HubError::NotFound
        );
        assert_eq!(
            map_http_error(ureq::Error::StatusCode(429)),
            HubError::RateLimited
        );
        assert_eq!(
            map_http_error(ureq::Error::BodyExceedsLimit(MAX_RESPONSE_BYTES)),
            HubError::ResponseTooLarge {
                limit_bytes: MAX_RESPONSE_BYTES
            }
        );
    }
}
