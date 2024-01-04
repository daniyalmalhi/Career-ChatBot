/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primaryBlue: '#164D5F',
        primaryWhite: '#FDFDFD'
      },
      fontFamily: {
        poppins: ['Poppins'],
        inter: ['Inter']
      }
    },
  },
  plugins: [],
}