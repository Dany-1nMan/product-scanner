require('dotenv').config();
const Vision = require('@google-cloud/vision');

(async () => {
  try {
    const client = process.env.GOOGLE_APPLICATION_CREDENTIALS
      ? new Vision.ImageAnnotatorClient({ keyFilename: process.env.GOOGLE_APPLICATION_CREDENTIALS })
      : new Vision.ImageAnnotatorClient();

    console.log('Client OK, project inferred.');
    // Simple empty call just to test auth path:
    await client.getProjectId(); // throws if creds invalid
    console.log('Project ID:', await client.getProjectId());
  } catch (e) {
    console.error('Auth test error:', e.message);
    console.error(e);
  }
})();
