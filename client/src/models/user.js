import {Schema, model, models} from 'mongoose';

const UserSchema = new Schema({
    email: {
        type: String,
        unique: true,
        required: [true, 'Please provide an email address'],
    },
    password: {
        type: String,
    },
    provider: {
        type: String,
    },
    username: {
        type: String,
        match: [/^(?=.{8,20}$)(?![_.])(?!.*[_.]{2})[a-zA-Z0-9._]+(?<![_.])$/,'"Username invalid, it should contain 8-20 alphanumeric letters and be unique!"']
    },
    image:{
        type: String,
    },
    type: {
        type: String
    },
    date: {
        type: Date
    }
});

const User = models.User || model('User', UserSchema);
export default User;