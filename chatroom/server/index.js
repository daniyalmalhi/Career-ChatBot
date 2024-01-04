const express = require('express');
const app = express()
const http = require('http')
const {Server}  = require('socket.io')
const cors = require('cors')
const dotenv = require("dotenv").config();
const connectDB = require("./utils/database");
connectDB()

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use("/api/users", require("./routes/UserRoutes"));


const server = http.createServer(app)

const io = new Server(server, {
    cors: {
        origin: 'http://localhost:3001',
        methods: ['GET', 'POST']
    }
})

io.on('connection', (socket) => {
    console.log(`User Connected: ${socket.id}`)
    socket.on("join_room", (data) => {
        console.log("index.js",data)
        socket.join(data);
      });
    
    socket.on('send_message', (data) => {
        console.log("send_message",data)
        socket.to(data.room).emit('receive_message', data)
    })
})
server.listen(3005, () => {
    console.log('Server is listening on *:3005')
})