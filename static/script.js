// Function to predict churn
async function predictChurn() {
  try {
      // Get form values
      const tenure = parseFloat(document.getElementById('tenure').value);
      const monthlyCharges = parseFloat(document.getElementById('monthly-charges').value);
      const totalCharges = parseFloat(document.getElementById('total-charges').value);
      const seniorCitizen = parseInt(document.getElementById('senior-citizen').value);
      const partner = parseInt(document.getElementById('partner').value);

      // Validate inputs
      if (isNaN(tenure) || isNaN(monthlyCharges) || isNaN(totalCharges)) {
          throw new Error("Invalid input values. Please enter valid numbers.");
      }

      // Prepare the payload
      const payload = {
          tenure,
          MonthlyCharges: monthlyCharges,
          TotalCharges: totalCharges,
          SeniorCitizen: seniorCitizen,
          Partner: partner
      };

      // Debugging: Print the payload to the console
      console.log("Sending Payload:", payload);

      // Send POST request to the API
      const response = await fetch('/predict', {  // Use relative path '/predict'
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
      });

      // Handle API response
      if (!response.ok) {
          const errorText = await response.text();  // Get detailed error message
          console.error('API Error:', errorText);
          throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();
      const resultDiv = document.getElementById('result');

      if (result.prediction === 0) {
          resultDiv.className = "result success";
          resultDiv.textContent = `Legitimate Customer! Churn Probability: ${result.churn_probability.toFixed(2)}%`;
      } else {
          resultDiv.className = "result error";
          resultDiv.textContent = `Potential Churn Detected! Churn Probability: ${result.churn_probability.toFixed(2)}%`;
      }
  } catch (error) {
      console.error('Error:', error);  // Log the error in the browser console
      const resultDiv = document.getElementById('result');
      resultDiv.className = "result error";
      resultDiv.textContent = "An error occurred while processing the request.";
  }
}