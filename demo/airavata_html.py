chat_html = """
<style>
    .chat-container {
        width: 100%;
        max-width: 600px;
        margin: auto;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        font-family: Arial, sans-serif;
        background-color: #f9f9f9;
        height: 400px;
        overflow-y: auto;
        position: relative;
    }
    .chat-message {
        margin-bottom: 10px;
    }
    .user-message {
        color: #007bff;
        font-weight: bold;
    }
    .bot-message {
        color: #333;
    }
    .chat-input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-top: 10px;
    }
    .chat-icon {
        position: fixed; /* Fixes the position relative to the viewport */
        top: 10px;
        right: 10px;
        width: 150px;
        height: auto;
        cursor: pointer;
        background-color: #fff;
        padding: 5px; /* Optional: add padding */
        border-radius: 5px; /* Optional: add rounded corners */
        z-index: 1000; /* Ensure the icon is on top of other elements */
    }
</style>
<div class="chat-container" id="chat-output">
    <img src="https://ai4bharat.github.io/airavata/static/images/cover.png" class="chat-icon" alt="Chatbot Icon">
    <div class="chat-message bot-message">
        <b>Airavata:</b> नमस्ते! मैं ऐरावत हूँ, आपका हिंदी और अंग्रेजी चैट सहायक। कृपया जारी रखने के लिए अपना प्रश्न हिंदी या अंग्रेजी में दर्ज करें। Hello! I'm Airavata, your Hindi and English Chat Assistant. Kindly enter your query in Hindi or English to continue.
    </div>
</div>
<input id="user-input" type="text" class="chat-input" placeholder="Type a message...">
"""





js_code = """
<script>
function handleInput() {
    var userInput = document.getElementById('user-input').value;
    var chatOutput = document.getElementById('chat-output');
    chatOutput.innerHTML += '<div class="chat-message bot-message"> <b>Airavata:</b> Processing...</div>';
    document.getElementById('user-input').value = '';
    google.colab.kernel.invokeFunction('process_input', [userInput], {});
}

document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        handleInput();
    }
});
</script>
"""
    




update_html = """
<script>
var chatOutput = document.getElementById('chat-output');
var messages = chatOutput.getElementsByClassName('bot-message');

// Remove the "Processing..." message
for (var i = 0; i < messages.length; i++) {{
    if (messages[i].textContent.includes('Processing...')) {{
        messages[i].remove();
        break;
    }}
}}

chatOutput.innerHTML += '<div class="chat-message user-message">User: {user_input}</div>';
chatOutput.innerHTML += '<div class="chat-message bot-message"><b>Airavata:</b> {response}</div>';
chatOutput.scrollTop = chatOutput.scrollHeight;  // Scroll to the bottom
</script>
"""