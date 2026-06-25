let conversation_history = []
const chatMessages = document.querySelector('.chat-messages');

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

    displayMessage('bot', 'Hello. I am your AI clinician assistant. How can I help you today?');
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

    function sendMessage() {
        const text = messageInput.value.trim();

        if (!text) {
            return;
        }

        appendMessage('user', text);
        messageInput.value = '';
        
        /*
        // TODO: Send user's message over to server 
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: text,
                history: conversation_history
            })
        })
        
        .then(response => response.json())
        .then(data => {
            // Temporary placeholder response until the API is connected.
            window.setTimeout(() => {
                appendMessage('bot', data['res']);
            }, 500);
        });
        */

        window.setTimeout(() => {
            appendMessage('bot', 'Thank you for your message! You will get your response when model is ready!')
        }, 3000)
    }

    function appendMessage(sender, text) {
        const messageRow = document.createElement('div');
        const messageBubble = document.createElement('div');


        messageRow.className = `message-row ${sender}`;
        messageBubble.className = 'message-bubble';
        messageBubble.textContent = text;

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