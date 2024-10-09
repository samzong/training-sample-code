package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

func main() {
        url := "http://127.0.0.1:11434/v1/chat/completions"
        messages := []map[string]string{
                {"role": "user", "content": "hello!"},
                {"role": "user", "content": "how are you?"},
        }
        bodyMap := map[string]interface{}{
                "model":   "llama3.2:latest",
                "messages": messages,
        }
        body, _ := json.Marshal(bodyMap)
        req, _ := http.NewRequest("POST", url, bytes.NewBuffer(body))
        req.Header.Add("Content-Type", "application/json")
        resp, err := http.DefaultClient.Do(req)
        if err != nil {
                fmt.Println("Error:", err)
                return
        }
        defer resp.Body.Close()
        var result map[string]interface{}
        json.NewDecoder(resp.Body).Decode(&result)
        fmt.Println(result)
}