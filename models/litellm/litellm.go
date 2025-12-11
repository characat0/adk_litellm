// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package lite_llm implements the [model.LLM] interface for LiteLLM proxy.
package litellm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"runtime"
	"strings"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// ClientConfig holds the configuration for the LiteLLM client.
type ClientConfig struct {
	// BaseURL is the LiteLLM proxy base URL (e.g., "http://localhost:4000")
	BaseURL string
	// APIKey is the API key for authentication
	APIKey string
	// HTTPClient is an optional custom HTTP client
	HTTPClient *http.Client
}

type liteLLMModel struct {
	config             *ClientConfig
	name               string
	versionHeaderValue string
	httpClient         *http.Client
}

// NewModel returns [model.LLM], backed by the LiteLLM proxy.
//
// It uses the provided context and configuration to initialize the client.
// The modelName specifies which model to target (e.g., "gpt-4", "claude-3-opus").
//
// An error is returned if the configuration is invalid.
func NewModel(ctx context.Context, modelName string, cfg *ClientConfig) (model.LLM, error) {
	if cfg == nil {
		return nil, fmt.Errorf("configuration is required")
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("BaseURL is required")
	}

	// Create header value once, when the model is created
	headerValue := fmt.Sprintf("google-adk/0.2.0 gl-go/%s",
		strings.TrimPrefix(runtime.Version(), "go"))

	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	return &liteLLMModel{
		name:               modelName,
		config:             cfg,
		versionHeaderValue: headerValue,
		httpClient:         httpClient,
	}, nil
}

func (m *liteLLMModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying model via LiteLLM proxy.
func (m *liteLLMModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	if stream {
		return m.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

// addHeaders sets the x-goog-api-client and user-agent headers
func (m *liteLLMModel) addHeaders(headers http.Header) {
	headers.Set("x-goog-api-client", m.versionHeaderValue)
	headers.Set("user-agent", m.versionHeaderValue)
	if m.config.APIKey != "" {
		headers.Set("Authorization", "Bearer "+m.config.APIKey)
	}
	headers.Set("Content-Type", "application/json")
}

// generate calls the model synchronously via LiteLLM proxy.
func (m *liteLLMModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	// Convert genai request to LiteLLM format
	liteLLMReq := m.convertToLiteLLMRequest(req)

	reqBody, err := json.Marshal(liteLLMReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := strings.TrimSuffix(m.config.BaseURL, "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	m.addHeaders(httpReq.Header)

	resp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var liteLLMResp liteLLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&liteLLMResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return m.convertToLLMResponse(&liteLLMResp), nil
}

// generateStream returns a stream of responses from the model via LiteLLM proxy.
func (m *liteLLMModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Convert genai request to LiteLLM format with streaming enabled
		liteLLMReq := m.convertToLiteLLMRequest(req)
		liteLLMReq.Stream = true

		reqBody, err := json.Marshal(liteLLMReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal request: %w", err))
			return
		}

		url := strings.TrimSuffix(m.config.BaseURL, "/") + "/chat/completions"
		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create request: %w", err))
			return
		}

		m.addHeaders(httpReq.Header)

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to call model: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body)))
			return
		}

		// Parse SSE stream
		reader := NewSSEReader(resp.Body)
		var lastYieldedResponse *model.LLMResponse

		for {
			event, err := reader.ReadEvent()
			if err == io.EOF {
				break
			}
			if err != nil {
				yield(nil, fmt.Errorf("failed to read stream: %w", err))
				return
			}

			if event == "[DONE]" {
				break
			}

			var chunk liteLLMStreamChunk
			if err := json.Unmarshal([]byte(event), &chunk); err != nil {
				continue // Skip malformed chunks
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]

			// Check if this is the final chunk (has a non-empty finish reason)
			isFinal := choice.FinishReason != nil && *choice.FinishReason != ""

			// Always yield if we have content or if it's the final chunk
			if choice.Delta.Content != "" || isFinal {
				llmResp := &model.LLMResponse{
					Partial: false,
				}

				// Add content if present
				if choice.Delta.Content != "" {
					llmResp.Content = &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							genai.NewPartFromText(choice.Delta.Content),
						},
					}
				}

				// If this is the final chunk, set finish reason
				if isFinal {
					llmResp.FinishReason = m.convertFinishReason(*choice.FinishReason)
					llmResp.TurnComplete = true
				}

				if !yield(llmResp, nil) {
					return // Consumer stopped
				}

				lastYieldedResponse = llmResp

				// If we already yielded the final chunk, we're done
				if isFinal {
					break
				}
			}
		}

		// Ensure the last yielded response is marked as final
		if lastYieldedResponse != nil && lastYieldedResponse.Partial {
			finalResp := &model.LLMResponse{
				Partial:      false,
				TurnComplete: true,
				FinishReason: genai.FinishReasonStop,
			}
			yield(finalResp, nil)
		}
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *liteLLMModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

type liteLLMResponseFormat struct {
	Type       string                 `json:"type"`
	JSONSchema map[string]interface{} `json:"json_schema"`
	Strict     bool                   `json:"strict"`
}

// LiteLLM request/response structures
type liteLLMRequest struct {
	Model          string                `json:"model"`
	Messages       []liteLLMMessage      `json:"messages"`
	Temperature    *float64              `json:"temperature,omitempty"`
	MaxTokens      *int                  `json:"max_tokens,omitempty"`
	Stream         bool                  `json:"stream,omitempty"`
	Tools          []liteLLMTool         `json:"tools,omitempty"`
	ToolChoice     interface{}           `json:"tool_choice,omitempty"`
	ResponseFormat liteLLMResponseFormat `json:"response_format,omitempty"`
}

type liteLLMMessage struct {
	Role       string            `json:"role"`
	Content    string            `json:"content,omitempty"`
	ToolCalls  []liteLLMToolCall `json:"tool_calls,omitempty"`
	ToolCallID string            `json:"tool_call_id,omitempty"`
}

type liteLLMTool struct {
	Type     string              `json:"type"`
	Function liteLLMFunctionDecl `json:"function"`
}

type liteLLMFunctionDecl struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type liteLLMToolCall struct {
	ID       string              `json:"id"`
	Type     string              `json:"type"`
	Function liteLLMFunctionCall `json:"function"`
}

type liteLLMFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type liteLLMResponse struct {
	ID      string          `json:"id"`
	Object  string          `json:"object"`
	Created int64           `json:"created"`
	Model   string          `json:"model"`
	Choices []liteLLMChoice `json:"choices"`
	Usage   liteLLMUsage    `json:"usage"`
}

type liteLLMChoice struct {
	Index        int            `json:"index"`
	Message      liteLLMMessage `json:"message"`
	FinishReason string         `json:"finish_reason"`
}

type liteLLMUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type liteLLMStreamChunk struct {
	ID      string                `json:"id"`
	Object  string                `json:"object"`
	Created int64                 `json:"created"`
	Model   string                `json:"model"`
	Choices []liteLLMStreamChoice `json:"choices"`
}

type liteLLMStreamChoice struct {
	Index        int          `json:"index"`
	Delta        liteLLMDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type liteLLMDelta struct {
	Role      string            `json:"role,omitempty"`
	Content   string            `json:"content,omitempty"`
	ToolCalls []liteLLMToolCall `json:"tool_calls,omitempty"`
}

// convertToLiteLLMRequest converts a model.LLMRequest to LiteLLM format
func (m *liteLLMModel) convertToLiteLLMRequest(req *model.LLMRequest) *liteLLMRequest {
	liteLLMReq := &liteLLMRequest{
		Model:    m.name,
		Messages: make([]liteLLMMessage, 0),
	}

	// Convert system instruction to system message
	if req.Config != nil && req.Config.SystemInstruction != nil {
		systemContent := m.extractTextFromParts(req.Config.SystemInstruction.Parts)
		if systemContent != "" {
			liteLLMReq.Messages = append(liteLLMReq.Messages, liteLLMMessage{
				Role:    "system",
				Content: systemContent,
			})
		}
	}

	// Convert contents to messages
	for _, content := range req.Contents {
		if content == nil {
			continue
		}

		role := content.Role
		if role == "model" {
			role = "assistant"
		}

		textContent := m.extractTextFromParts(content.Parts)
		liteLLMReq.Messages = append(liteLLMReq.Messages, liteLLMMessage{
			Role:    role,
			Content: textContent,
		})
	}

	// Add configuration parameters
	if req.Config != nil {
		if req.Config.Temperature != nil {
			temp := float64(*req.Config.Temperature)
			liteLLMReq.Temperature = &temp
		}
		if req.Config.MaxOutputTokens != 0 {
			maxTokens := int(req.Config.MaxOutputTokens)
			liteLLMReq.MaxTokens = &maxTokens
		}
	}

	return liteLLMReq
}

// extractTextFromParts extracts text content from genai Parts
func (m *liteLLMModel) extractTextFromParts(parts []*genai.Part) string {
	var texts []string
	for _, part := range parts {
		if part == nil {
			continue
		}
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// convertToLLMResponse converts a LiteLLM response to model.LLMResponse
func (m *liteLLMModel) convertToLLMResponse(resp *liteLLMResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{}
	}

	choice := resp.Choices[0]

	return &model.LLMResponse{
		Content: &genai.Content{
			Role: "model",
			Parts: []*genai.Part{
				genai.NewPartFromText(choice.Message.Content),
			},
		},
		FinishReason: m.convertFinishReason(choice.FinishReason),
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		},
		TurnComplete: choice.FinishReason == "stop",
	}
}

// convertFinishReason converts LiteLLM finish reason to genai format
func (m *liteLLMModel) convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}

// SSEReader reads Server-Sent Events from an io.Reader
type SSEReader struct {
	reader *bytes.Reader
	buffer []byte
}

// NewSSEReader creates a new SSE reader
func NewSSEReader(r io.Reader) *SSEReader {
	data, _ := io.ReadAll(r)
	return &SSEReader{
		reader: bytes.NewReader(data),
		buffer: data,
	}
}

// ReadEvent reads the next SSE event
func (r *SSEReader) ReadEvent() (string, error) {
	for {
		line, err := r.readLine()
		if err != nil {
			return "", err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			return data, nil
		}
	}
}

func (r *SSEReader) readLine() (string, error) {
	var line []byte
	for {
		b, err := r.reader.ReadByte()
		if err != nil {
			if len(line) > 0 {
				return string(line), nil
			}
			return "", err
		}
		if b == '\n' {
			return string(line), nil
		}
		if b != '\r' {
			line = append(line, b)
		}
	}
}
