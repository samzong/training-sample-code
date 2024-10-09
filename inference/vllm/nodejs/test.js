const OpenAI = require('openai');

const openai = new OpenAI({
  baseURL: 'http://127.0.0.1:11434/v1',
  apiKey: 'custom',
});

async function getData() {
  try {
    const chatCompletion = await openai.chat.completions.create({
      model: 'llama3.2:latest',
      messages: [
        { role: 'user', content: 'hello!' },
        { role: 'user', content: 'how are you?' },
      ],
    });

    console.log(chatCompletion.choices[0].message.content);
  } catch (error) {
    if (error instanceof OpenAI.APIError) {
      console.error('API Error:', error.status, error.message);
      console.error('Error details:', error.error);
    } else {
      console.error('Unexpected error:', error);
    }
  }
}

getData();