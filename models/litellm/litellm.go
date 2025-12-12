package litellm

import (
	"context"
	"encoding/json"
	"iter"
	"net/url"
	"strings"

	"github.com/andrejsstepanovs/go-litellm/client"
	"github.com/andrejsstepanovs/go-litellm/common"
	"github.com/andrejsstepanovs/go-litellm/conf/connections/litellm"
	"github.com/andrejsstepanovs/go-litellm/models"
	"github.com/andrejsstepanovs/go-litellm/request"
	"github.com/google/jsonschema-go/jsonschema"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type Target litellm.Target

type LiteLLMModel struct {
	*client.Litellm
	models.ModelMeta
	DefaultTemperature float32
}

type ClientConfig struct {
	BaseURL       string
	APIKey        string
	Temperature   float32
	SystemTimeout Target
	LLMTimeout    Target
	MCPTimeout    Target
}

func NewModel(ctx context.Context, modelName string, cfg *ClientConfig) (*LiteLLMModel, error) {
	clientCfg := client.Config{
		APIKey:      cfg.APIKey,
		Temperature: cfg.Temperature,
	}
	baseURL, err := url.Parse(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	clientConn := litellm.Connection{
		URL: *baseURL,
		Targets: litellm.Targets{
			System: litellm.Target(cfg.SystemTimeout),
			LLM:    litellm.Target(cfg.LLMTimeout),
			MCP:    litellm.Target(cfg.MCPTimeout),
		},
	}
	ai, err := client.New(clientCfg, clientConn)
	if err != nil {
		return nil, err
	}
	model, err := ai.Model(ctx, models.ModelID(modelName))
	if err != nil {
		return nil, err
	}
	return &LiteLLMModel{Litellm: ai, ModelMeta: model, DefaultTemperature: cfg.Temperature}, err
}

func (m *LiteLLMModel) Name() string {
	return string(m.ModelMeta.ModelId)
}

func extractText(parts []*genai.Part) string {
	var texts []string
	for _, part := range parts {
		texts = append(texts, string(part.Text))
	}
	return strings.Join(texts, " ")
}

func schemaToLLMCallToolFunctionProperty(schema *jsonschema.Schema) request.LLMCallToolFunctionProperty {
	return request.LLMCallToolFunctionProperty{
		Description: schema.Description,
		Type:        schema.Type,
		Enum:        nil,
		Format:      schema.Format,
		Example:     "",
	}
}

func schemaToFunctionParameters(schema *jsonschema.Schema) *request.LLMCallToolFunctionParameters {
	properties := make(map[string]request.LLMCallToolFunctionProperty)
	for name, prop := range schema.Properties {
		properties[name] = schemaToLLMCallToolFunctionProperty(prop)
	}

	return &request.LLMCallToolFunctionParameters{
		Type:       schema.Type,
		Properties: properties,
		Required:   schema.Required,
	}
}

func toJSONSchema(s *genai.Schema) map[string]interface{} {
	str, _ := json.Marshal(s)
	var schema map[string]interface{}
	_ = json.Unmarshal(str, &schema)
	schema["additionalProperties"] = false
	return schema
}

func toLiteLLMCompletionRequest(m models.ModelMeta, req *model.LLMRequest, stream bool, defaultTemperature float32) *request.Request {
	var messages []request.Message

	if req.Config.SystemInstruction != nil {
		messages = append(messages, request.SystemMessageSimple(extractText(req.Config.SystemInstruction.Parts)))
	}

	for _, content := range req.Contents {
		if content == nil {
			continue
		}
		txt := extractText(content.Parts)
		switch content.Role {
		case genai.RoleUser:
			messages = append(messages, request.UserMessageSimple(txt))
		case genai.RoleModel:
			messages = append(messages, request.AssistantMessageSimple(txt))
		}
	}

	var tools []request.LLMCallTool
	for _, tool := range req.Config.Tools {
		for _, funcTool := range tool.FunctionDeclarations {
			schema := funcTool.ParametersJsonSchema.(*jsonschema.Schema)
			addTool := request.LLMCallTool{
				Type: request.FunctionToolType,
				Function: &request.LLMCallToolFunction{
					Name:        funcTool.Name,
					Description: funcTool.Description,
					Parameters:  schemaToFunctionParameters(schema),
				},
			}
			tools = append(tools, addTool)
		}
	}
	liteReq := request.NewCompletionRequest(m, messages, tools, req.Config.Temperature, defaultTemperature)

	if req.Config.ResponseSchema != nil {
		liteReq.SetJSONSchema(request.JSONSchema{
			Name:   req.Config.ResponseSchema.Title,
			Schema: toJSONSchema(req.Config.ResponseSchema),
			Strict: true,
		})
	}
	liteReq.Stream = false

	return liteReq
}

func convertFinishReason(reason string) genai.FinishReason {
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

func NewPartsFromToolCalls(calls []common.ToolCall) []*genai.Part {
	var parts []*genai.Part
	for _, call := range calls {
		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				ID:   call.ID,
				Args: call.Function.Arguments,
				Name: call.Function.Name,
			},
		}
		parts = append(parts, part)
	}
	return parts
}

func (m *LiteLLMModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	litellmRequest := toLiteLLMCompletionRequest(m.ModelMeta, req, stream, m.DefaultTemperature)
	return func(yield func(*model.LLMResponse, error) bool) {
		res, err := m.Litellm.Completion(ctx, litellmRequest)
		if err != nil {
			yield(nil, err)
			return
		}
		msg := res.Message()

		var parts []*genai.Part
		if msg.Content != "" {
			parts = append(parts, genai.NewPartFromText(msg.Content))
		}

		parts = append(parts, NewPartsFromToolCalls(msg.ToolCalls)...)

		out := &model.LLMResponse{
			Content: &genai.Content{
				Role:  genai.RoleModel,
				Parts: parts,
			},
			FinishReason: convertFinishReason(string(res.Choice().FinishReason)),
			TurnComplete: true,
		}
		if !yield(out, nil) {
			return
		}
	}
}
