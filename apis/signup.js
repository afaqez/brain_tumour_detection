document.getElementById('signupForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const firstName = document.getElementById('firstName').value;
    const lastName = document.getElementById('lastName').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const username = `${firstName}${lastName}`;

    const data = {
        username: username,
        email: email,
        password: password,
        role: "user"
    };

    fetch('https://fitnessapp-666y.onrender.com/api/signup', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.TOKEN) {
            localStorage.setItem('TOKEN', data.TOKEN);
            updateUIForLoggedInUser();
            window.location.href = '../template/home.html'; 
        } else {
            alert('Signup failed: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function updateUIForLoggedInUser() {
    document.getElementById('loginLink').style.display = 'none';
    document.getElementById('logoutLink').style.display = 'inline';
    document.getElementById('logoutLink').addEventListener('click', function(event) {
        localStorage.removeItem('TOKEN');
        document.getElementById('loginLink').style.display = 'inline';
        document.getElementById('logoutLink').style.display = 'none';
        window.location.href = '../template/login.html'; 
    });
}

window.onload = function() {
    if (localStorage.getItem('TOKEN')) {
        updateUIForLoggedInUser();
    }
};
