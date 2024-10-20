import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
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
        "give a bunch of one liners that will calm the reader down in the ultimate attempt of them to not get angry and angrier\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "## One-Liners to Calm Down:\n\n**General:**\n\n* Breathe in, breathe out. It's not worth the stress.\n* Take a moment, let the anger pass.\n* It's okay to feel frustrated, but don't let it control you.\n* This too shall pass.\n* What's done is done, focus on what you can control.\n\n**Humor:**\n\n* Anger is a hot emotion, let it simmer down.\n* You're not a volcano, don't erupt.\n* Relax, it's just a bad day, not a bad life.\n* Take a deep breath and count to ten... or twenty if you need it.\n* Life is too short to be angry all the time.\n\n**Perspective:**\n\n* Is this really worth getting worked up about?\n* This situation isn't the end of the world.\n* Look at the bigger picture, it's all temporary.\n* Everyone makes mistakes, including you.\n* Focus on what's good in your life, not what's bad.\n\n**Actionable:**\n\n* Take a walk, get some fresh air.\n* Do something you enjoy, a hobby or activity.\n* Talk to someone you trust, get it off your chest.\n* Practice mindfulness, focus on the present moment.\n* Let go of the need to control everything.\n\nRemember, these are just starting points.  The best one-liner for someone will depend on their individual personality and the situation they are facing.  It's also important to be genuine and compassionate when offering these words of advice. \n",
      ],
    },
    {
      "role": "user",
      "parts": [
        "try not to mention anything about hobbies or doing other hobbies\n",
      ],
    },
    {
      "role": "model",
      "parts": [
        "Here are some one-liners to help someone calm down, avoiding mentions of hobbies or other activities:\n\n**General:**\n\n* Take a deep breath, it's okay to feel overwhelmed.\n* Let the anger pass, it's just a wave, not a tidal wave.\n* Focus on the present moment, this feeling will pass.\n* You are strong, you can handle this.\n* It's okay to take a break, give yourself time to cool down.\n\n**Humor:**\n\n* This is just a blip on the radar of life, relax.\n* You're not a pressure cooker, don't let the steam build up.\n* Anger is like a hot potato, don't hold on too long.\n* Take a moment, put things in perspective.\n* This is not the apocalypse, it's just a bad day.\n\n**Perspective:**\n\n* Is this really worth getting worked up about?\n* Everyone makes mistakes, it's part of being human.\n* You are not alone, we all have bad days.\n* Take a step back, the situation isn't as bad as it seems.\n* This too shall pass, you will get through this.\n\n**Actionable:**\n\n* Take a walk, get some fresh air and clear your head.\n* Close your eyes, take some deep breaths, and visualize a peaceful scene.\n* Listen to calming music, let it soothe your nerves.\n* Talk to someone you trust, getting it off your chest can help.\n* Practice mindfulness, focus on the present moment and let go of thoughts.\n\nRemember, these are just starting points. It's important to tailor your response to the specific person and situation. \n",
      ],
    },
  ]
)

response = chat_session.send_message("INSERT_INPUT_HERE")

print(response.text)