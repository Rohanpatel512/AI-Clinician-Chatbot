let conversation_history = []
const chatMessages = document.querySelector('.chat-messages');
const thinkingIndicator = document.querySelector('#thinking-indicator');
const API_BASED_URL = window.APP_CONFIG?.API_BASED_URL ?? "";

window.onload = () => {
    const greeting = document.getElementById('greeting');
    const hour = new Date().getHours();

    if (hour < 12) {
        greeting.textContent = typeGreet('Good morning');
    } else if (hour < 18) {
        greeting.textContent = typeGreet('Good afternoon');
    } else {
        greeting.textContent = typeGreet('Good evening');
    }

    displayMessage('Assistant', 'Hello. I am your AI clinician assistant. How can I help you today?');
    setupChat();
};

function typeGreet(greet) {
    let index = 0;
    const fullText = `${greet}`;

    const interval = setInterval(() => {
        if (index < fullText.length) {
            document.getElementById('greeting').textContent += fullText.charAt(index);
            index++;
        } else {
            clearInterval(interval);
        }
    }, 100);
}

function setupChat() {
    const messageInput = document.getElementById('messages');
    const sendButton = document.getElementById('send-btn');

    sendButton.addEventListener('click', sendMessage);

    messageInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const text = messageInput.value.trim();

        if (!text) {
            return;
        }

        displayMessage('Patient', text);
        messageInput.value = '';

        // Toggle-on thinking mode
        toggleThinking(true);

        try {
            const response = await fetch(`${API_BASED_URL}/chat/send_message`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: text, conversation_history: conversation_history}),
            });

            toggleThinking(false);
            const data = await response.json()


            appendMessage('Assistant', data['response'])
        } catch(error) {
            console.error('Error sending message:', error);
        }
    }

    function appendMessage(sender, text) {
        const messageRow = document.createElement('div');
        const messageBubble = document.createElement('div');


        messageRow.className = `message-row ${sender}`;
        messageBubble.className = 'message-bubble';
        type(text, messageBubble);

        messageRow.appendChild(messageBubble);
        chatMessages.appendChild(messageRow);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        addToHistory(sender, text)
    }
}

function addToHistory(sender, text) {
    conversation_history.push({role: sender, content: text});
    var length = conversation_history.length;
    if (length == 7) {
        conversation_history = conversation_history.slice(length - 3, length);
    }
    console.log(conversation_history)
    return conversation_history
}

function displayMessage(sender, text) {
    const messageRow = document.createElement('div');
    const messageBubble = document.createElement('div');


    messageRow.className = `message-row ${sender}`;
    messageBubble.className = 'message-bubble';
    messageBubble.textContent = text;
    messageRow.appendChild(messageBubble);
    chatMessages.appendChild(messageRow);
    chatMessages.scrollTop = chatMessages.scrollHeight;

}       

function toggleThinking(isThinking) {

    console.log("is thinking function on...");

    if(isThinking) {
        // Unhide thinking indicator
        thinkingIndicator.classList.remove("hidden");
    } else {
        thinkingIndicator.classList.add("hidden");
    }

}

function type(text, messageBubble) {

    return new Promise((resolve) => {
        messageBubble.textContent = "";
        let index = 0;

        const interval = setInterval(() => {
            if (index < text.length) {
                messageBubble.textContent += text[index];
                index++;
            } else {
                clearInterval(interval);
                resolve();
            }
        }, 50);
    });

}