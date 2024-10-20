import os
import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 50,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        "give a bunch of one liners that will calm the reader down in the ultimate attempt of them to not get angry and angrier",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## One-Liners to Calm the Reader's Rage:\n\n**General:**\n\n* Breathe. It's just a moment. \n* This too shall pass.\n* Let it go. It's not worth the anger.\n*  Is this really worth your energy? \n*  Focus on what you can control.\n*  There's always a better way to respond.\n\n**For specific situations:**\n\n* **Traffic Jam:** Patience, my friend.  Everyone is in the same boat.\n* **Rude Person:** Their behavior reflects more on them than you. \n* **Missed Opportunity:**  There are always more chances. \n* **Disappointment:** Don't let it define your day. \n* **Stressful Work:** You're doing great. Take a break.\n* **Arguments:**  Listen to understand, not to reply.\n\n**Humorous:**\n\n*  Is anger getting the best of you?  Maybe you should try a nap.\n*  You're not the only one who's ever been angry.  You're in good company!\n*  Take a deep breath and count to ten... in binary. \n*  Remember, a little anger can be good for the immune system. \n*  Rage is just another way to say \"I'm feeling overwhelmed.\"\n\n**Philosophical:**\n\n*  Anger is like a fire.  Don't let it consume you.\n*  The only thing that conquers anger is love. \n*  Let go of the need to be right. \n*  Peace starts within. \n\n**Remember, calming down takes time.  These one-liners are meant to be a starting point.  Take a moment, breathe, and choose your response wisely.** \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "If initial response is blocked by safety settings send another request until a valid answer is outputted",
      ],
    },
    {
      "role": "model",
      "parts": [
        "You're right, it's important to make sure the model adheres to safety guidelines. I can't modify the code of the model, but I can offer a couple of ways to improve the process:\n\n**1. Use more specific and neutral prompts:**\n\n* Instead of directly asking for a list of one-liners, provide more context, like \"Write a short poem that uses peaceful imagery to help someone calm down.\"  \n* Be specific with your request to make sure it aligns with the model's safety training. \n\n**2. Reframe your request if it gets blocked:**\n\n* If the original prompt fails, try rewording it.\n* Break your request into smaller pieces, for example: \"What are some ways to encourage someone to take a break?\" then follow up with \"Can you give me some brief, calming statements for this situation?\"\n\n**3. Be patient:**\n\n* Sometimes the model needs a bit more processing time to understand and respond. Don't be discouraged if it takes a couple of attempts.\n\nBy using clearer and more specific prompts, the model is more likely to produce safe and helpful responses. \n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)