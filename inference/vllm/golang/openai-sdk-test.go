package main

import (
	"context"
	"net/http"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func main() {
	client := openai.NewClient(
		option.WithAPIKey("sk-xxx"),
		option.WithBaseURL("http://127.0.0.1:11434"),
		option.WithMiddleware(func(r *http.Request, mn option.MiddlewareNext) (*http.Response, error) {
			r.URL.Path = "/v1" + r.URL.Path
			return mn(r)
		}),
	)
	ctx := context.Background()
	question := "say this is a test"

	print("> ")
	println(question)
	println()

	completion, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(question),
		}),
		Seed:  openai.Int(1),
		Model: openai.F("Qwen/Qwen2.5-72B-Instruct"),
	})
	if err != nil {
		panic(err)
	}

	println(completion.Choices[0].Message.Content)
}
