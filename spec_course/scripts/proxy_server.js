const express = require("express");
const axios = require("axios");

const app = express();
const PORT = 9000;

app.use(express.json());
const logger = {
  info: (...args) => console.log("[INFO]", ...args),
  error: (...args) => console.error("[ERROR]", ...args),
};

app.post(process.env.API_ROUTE, async (req, res) => {
  const axiosConfig = {
    method: "post",
    url: process.env.ENDPOINT_URL + process.env.API_ROUTE,
    headers: {
      "Content-Type": "application/json",
    },
    data: req.body,
  };

  try {
    const realServerResponse = await axios(axiosConfig);
    const responseData = realServerResponse.data;
    res.json(responseData);
  } catch (error) {
    logger.error("Error in non-streaming request:", error);
    res.status(500).send("Internal Server Error");
  }
});

app.listen(PORT, () => {
  logger.info("Proxy server listening on port", PORT);
});
