const express = require('express')
const router = express.Router()
const {
  checkUser,
  getUsers,
  generateChatRoom,
  updateConvo,
  getRegUsers,
  getConvo
} = require('../controllers/userController')

router.post('/check', checkUser)
router.get('/', getUsers)
router.post('/room', generateChatRoom)
router.post('/convo', updateConvo)
router.post('/getRegUsers', getRegUsers)
router.post('/getConvo', getConvo)

module.exports = router
