/** @type {import('next').NextConfig} */
const helmet = require("helmet");
const nextConfig = {
  images: {
    domains: [
      "lh3.googleusercontent.com",
      "avatars.githubusercontent.com",
      "platform-lookaside.fbsbx.com",
    ],
  },
  async headers() {
    return [
      {
        source: "/(.*)", // You can specify a more specific source pattern if needed
        headers: [
          {
            key: "Content-Security-Policy",
            value: "frame-ancestors *", // Adjust this based on your requirements
          },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
