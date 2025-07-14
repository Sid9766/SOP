console.log("📘 script.js loaded");
let sopUploaded = false;
let datasetUploaded = false;
let currentAnalysis = { violations: [] };

// SOP Upload Handler
document.getElementById('sopForm').addEventListener('submit', async (e) => {
  console.log("🖱️  sopForm submit handler fired");
  e.preventDefault();
  const formData = new FormData();
  const file = document.getElementById('sopFile').files[0];
  
  if (!file) return;
  
  formData.append('file', file);
  updateStatus('sopStatus', 'pending', 'Uploading SOP...');
  console.log("🚀 About to POST /upload-sop/ with FormData:", formData.get('file'));
  try {
    const response = await fetch('http://127.0.0.1:8000/upload-sop/', {
      method: 'POST',
      body: formData
    });
    console.log("🛰️ fetch() returned, status =", response.status);
    const result = await response.json();
    
    if (result.success) {
      sopUploaded = true;
      updateStatus('sopStatus', 'success', `SOP uploaded (${result.total_items} items)`);
      displaySOPSummary(result.sop_data);
      checkAnalyzeButton();
    } else {
      updateStatus('sopStatus', 'error', 'Upload failed: ' + result.error);
    }
  } catch (error) {
    updateStatus('sopStatus', 'error', 'Upload failed');
    console.error('SOP upload error:', error);
  }
});

// Dataset Upload Handler
document.getElementById('datasetForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData();
  const file = document.getElementById('datasetFile').files[0];
  
  if (!file) return;
  
  formData.append('file', file);
  updateStatus('datasetStatus', 'pending', 'Uploading dataset...');
  
  try {
    const response = await fetch('http://127.0.0.1:8000/upload-dataset/', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
      datasetUploaded = true;
      updateStatus(
        'datasetStatus',
        'success',
        `Dataset uploaded (${result.dataset_info.rows} rows, ${result.dataset_info.columns} columns)`
      );
      checkAnalyzeButton();
    } else {
      updateStatus('datasetStatus', 'error', 'Upload failed: ' + result.error);
    }
  } catch (error) {
    updateStatus('datasetStatus', 'error', 'Upload failed');
    console.error('Dataset upload error:', error);
  }
});

// Analyze Button Handler
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  if (!sopUploaded || !datasetUploaded) return;
  
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
  
  try {
    const response = await fetch('http://127.0.0.1:8000/analyze-violations/', {
      method: 'POST'
    });
    
    const result = await response.json();
    
    if (result.success) {
      currentAnalysis = result.analysis;
      displayResults(result.analysis);
    } else {
      alert('Analysis failed: ' + result.error);
    }
  } catch (error) {
    alert('Analysis failed');
    console.error('Analysis error:', error);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-search me-2"></i>Analyze Violations';
  }
});

function updateStatus(elementId, status, message) {
  const element = document.getElementById(elementId);
  element.innerHTML = `<span class="status-indicator status-${status}"></span>${message}`;
}

function checkAnalyzeButton() {
  document.getElementById('analyzeBtn').disabled = !(sopUploaded && datasetUploaded);
}

function displaySOPSummary(sopData) {
  const summaryElement = document.getElementById('sopSummary');
  const contentElement = document.getElementById('sopContent');
  
  let content = '';
  
  if (sopData.steps && sopData.steps.length > 0) {
    content += '<h6><i class="fas fa-list-ol me-2"></i>Procedural Steps:</h6><ul class="list-group list-group-flush mb-3">';
    sopData.steps.forEach((step, index) => {
      content += `<li class="list-group-item">${index + 1}. ${step}</li>`;
    });
    content += '</ul>';
  }
  
  if (sopData.rules && sopData.rules.length > 0) {
    content += '<h6><i class="fas fa-gavel me-2"></i>Rules:</h6><ul class="list-group list-group-flush mb-3">';
    sopData.rules.forEach((rule, index) => {
      content += `<li class="list-group-item">${index + 1}. ${rule}</li>`;
    });
    content += '</ul>';
  }
  
  if (sopData.requirements && sopData.requirements.length > 0) {
    content += '<h6><i class="fas fa-check-circle me-2"></i>Requirements:</h6><ul class="list-group list-group-flush mb-3">';
    sopData.requirements.forEach((req, index) => {
      content += `<li class="list-group-item">${index + 1}. ${req}</li>`;
    });
    content += '</ul>';
  }
  
  // If no structured data, show combined items
  if (!content && sopData.steps) {
    content += '<h6><i class="fas fa-list me-2"></i>Extracted Items:</h6><ul class="list-group list-group-flush">';
    sopData.steps.forEach((item, index) => {
      content += `<li class="list-group-item">${index + 1}. ${item}</li>`;
    });
    content += '</ul>';
  }
  
  contentElement.innerHTML = content;
  summaryElement.style.display = 'block';
}

function displayResults(analysis) {
  const resultsSection = document.getElementById('resultsSection');
  
  // Update metrics
  document.getElementById('totalViolations').textContent = analysis.summary.total_violations;
  document.getElementById('complianceRate').textContent = `${analysis.summary.compliance_rate.toFixed(1)}%`;
  document.getElementById('criticalViolations').textContent = analysis.summary.critical;
  document.getElementById('compliantRows').textContent = analysis.summary.compliant_rows;
  
  // Create charts
  createSeverityChart(analysis.summary);
  createComplianceChart(analysis.summary);
  
  // Display violations list
  displayViolations(analysis.violations);
  
  resultsSection.style.display = 'block';
}

function createSeverityChart(summary) {
  const ctx = document.getElementById('severityChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'],
      datasets: [{
        data: [summary.critical, summary.high, summary.medium, summary.low],
        backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { 
          display: true, 
          text: 'VIOLATIONS BY SEVERITY',
          font: {
            size: 15
          }
        },
        legend: {
          labels: {
            font: {
              size: 13
            }
          }
        }
      }
    }
  });
}

function createComplianceChart(summary) {
  const ctx = document.getElementById('complianceChart').getContext('2d');
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: ['COMPLIANT', 'NON-COMPLIANT'],
      datasets: [{
        data: [summary.compliant_rows, summary.total_violations],
        backgroundColor: ['#28a745', '#dc3545']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: { 
          display: true, 
          text: 'OVERALL COMPLIANCE',
          font: {
            size: 15
          }
        },
        legend: {
          labels: {
            font: {
              size: 13
            }
          }
        }
      }
    }
  });
}

function displayViolations(violations) {
  currentAnalysis.violations = violations;
  const container = document.getElementById('violationsList');
  
  if (violations.length === 0) {
    container.innerHTML = '<div class="alert alert-success">No violations found! Dataset is fully compliant.</div>';
    return;
  }
  
  let html = '';
  violations.forEach((violation, index) => {
    html += `
      <div class="violation-card card severity-${violation.severity.toLowerCase()} mb-3" id="violation-card-${index}">
        <div class="card-body">
          <div class="d-flex justify-content-between align-items-start">
            <div class="flex-grow-1">
              <h6 class="card-title">
                <span class="severity-badge badge bg-${getSeverityColor(violation.severity)} me-2">
                  ${violation.severity}
                </span>
                ${violation.rule_violated || violation.violation_type || 'Rule Violation'}
              </h6>
              <p class="card-text">${violation.description}</p>
              <small class="text-muted">
                <i class="fas fa-table-list me-1"></i>Row: ${violation.row_number}
                ${violation.column ? ` | Column: ${violation.column}` : ''}
                ${violation.field_name ? ` | Field: ${violation.field_name}` : ''}
              </small>
            </div>
            <button class="btn btn-sm btn-outline-primary details-btn" data-index="${index}">
              <i class="fas fa-eye me-1"></i>Details
            </button>
          </div>
        </div>
        <div class="violation-details-container" id="details-container-${index}" style="display: none;">
          <div class="violation-details-content p-3 border-top">
            <div class="d-flex justify-content-between align-items-center mb-2">
              <h6 class="mb-0"><i class="fas fa-info-circle me-2"></i>Detailed Analysis</h6>
              <small class="text-muted">Loading...</small>
            </div>
            <div class="details-content" id="details-content-${index}">
              <div class="text-center p-3">
                <i class="fas fa-spinner fa-spin me-2"></i>Loading detailed analysis...
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  });
  
  container.innerHTML = html;
  
  // Add event listeners to details buttons
  document.querySelectorAll('.details-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const index = parseInt(btn.dataset.index);
      await toggleViolationDetails(index, btn);
    });
  });
}

async function toggleViolationDetails(index, buttonElement) {
  const detailsContainer = document.getElementById(`details-container-${index}`);
  const detailsContent = document.getElementById(`details-content-${index}`);
  const isVisible = detailsContainer.style.display !== 'none';
  
  if (isVisible) {
    // Hide details
    detailsContainer.style.display = 'none';
    buttonElement.innerHTML = '<i class="fas fa-eye me-1"></i>Details';
    buttonElement.classList.remove('btn-outline-secondary');
    buttonElement.classList.add('btn-outline-primary');
  } else {
    // Show details
    detailsContainer.style.display = 'block';
    buttonElement.innerHTML = '<i class="fas fa-eye-slash me-1"></i>Hide Details';
    buttonElement.classList.remove('btn-outline-primary');
    buttonElement.classList.add('btn-outline-secondary');
    
    // Check if we need to load the details
    const contentDiv = detailsContent.querySelector('.details-content') || detailsContent;
    if (contentDiv.innerHTML.includes('Loading detailed analysis...')) {
      try {
        // Call the backend API to get detailed analysis
        const response = await fetch('http://127.0.0.1:8000/violation-details/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            violation_index: index
          })
        });
        
        const result = await response.json();
        
        if (result.success) {
          // Format and display the detailed explanation
          const formattedContent = formatDetailedExplanation(result.detailed_explanation);
          detailsContent.innerHTML = `
            <div class="details-content">
              ${formattedContent}
            </div>
          `;
        } else {
          detailsContent.innerHTML = `
            <div class="alert alert-warning">
              <i class="fas fa-exclamation-triangle me-2"></i>
              Unable to load detailed analysis: ${result.error || 'Unknown error'}
            </div>
          `;
        }
      } catch (error) {
        console.error('Error fetching violation details:', error);
        detailsContent.innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-times-circle me-2"></i>
            Error loading detailed analysis. Please try again.
          </div>
        `;
      }
    }
  }
}

function formatDetailedExplanation(explanation) {
  // Convert the text response to HTML with proper formatting
  if (!explanation) {
    return '<p>No detailed explanation available.</p>';
  }
  
  // Split by double newlines to get sections
  const sections = explanation.split('\n\n');
  let formattedHtml = '';
  
  sections.forEach(section => {
    const lines = section.split('\n');
    const firstLine = lines[0].trim();
    
    // Check if this is a header (contains ** or ###)
    if (firstLine.includes('**') || firstLine.startsWith('#')) {
      const headerText = firstLine.replace(/\*\*|###|##|#/g, '').trim();
      formattedHtml += `<h6 class="mt-3 mb-2"><i class="fas fa-chevron-right me-2"></i>${headerText}</h6>`;
      
      // Add remaining lines as content
      if (lines.length > 1) {
        const content = lines.slice(1).join('\n').trim();
        if (content) {
          formattedHtml += formatContent(content);
        }
      }
    } else {
      // Regular content
      formattedHtml += formatContent(section);
    }
  });
  
  return formattedHtml || '<p>No detailed explanation available.</p>';
}

function formatContent(content) {
  if (!content.trim()) return '';
  
  // Check if content has bullet points
  if (content.includes('- ') || content.includes('• ')) {
    const items = content.split('\n').filter(line => line.trim());
    let listHtml = '<ul class="list-unstyled">';
    
    items.forEach(item => {
      const cleanItem = item.replace(/^[-•]\s*/, '').trim();
      if (cleanItem) {
        listHtml += `<li class="mb-1"><i class="fas fa-circle text-primary me-2" style="font-size: 0.5em; vertical-align: middle;"></i>${cleanItem}</li>`;
      }
    });
    
    listHtml += '</ul>';
    return listHtml;
  } else {
    // Regular paragraph
    return `<p class="mb-2">${content.trim()}</p>`;
  }
}

function getSeverityColor(severity) {
  const map = { critical: 'danger', high: 'warning', medium: 'info', low: 'success' };
  return map[severity.toLowerCase()] || 'secondary';
}

function viewViolationDetails(index) {
  // This function is kept for backward compatibility but redirects to the new toggle function
  const button = document.querySelector(`[data-index="${index}"]`);
  if (button) {
    toggleViolationDetails(index, button);
  }
}

/* ---------------------------
   Chat Sidebar Functionality
---------------------------- */

const chatToggle    = document.getElementById('chat-toggle');
const chatSidebar   = document.getElementById('chat-sidebar');
const chatClose     = document.getElementById('chat-close');
const chatMessages  = document.getElementById('chat-messages');
const chatInput     = document.getElementById('chat-input-field');
const chatSendBtn   = document.getElementById('chat-send');

// Open chat
chatToggle.addEventListener('click', () => {
  chatSidebar.classList.add('open');
  chatInput.focus();
});

// Close chat
chatClose.addEventListener('click', () => {
  chatSidebar.classList.remove('open');
});

// Send a chat message
chatSendBtn.addEventListener('click', async () => {
  const question = chatInput.value.trim();
  if (!question) return;
  
  // Append user message
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-msg chat-msg-user mb-2';
  userDiv.textContent = question;
  chatMessages.appendChild(userDiv);
  chatInput.value = '';
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  // Fetch response
  try {
    const res = await fetch('http://127.0.0.1:8000/chat/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const { answer } = await res.json();
    
    const botDiv = document.createElement('div');
    botDiv.className = 'chat-msg chat-msg-bot mb-2';
    botDiv.textContent = answer || 'Sorry, no response.';
    chatMessages.appendChild(botDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  } catch (err) {
    console.error('Chat error:', err);
    const errDiv = document.createElement('div');
    errDiv.className = 'chat-msg chat-msg-bot mb-2';
    errDiv.textContent = 'Error fetching response.';
    chatMessages.appendChild(errDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
});

// Also send on Enter key
chatInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    chatSendBtn.click();
  }
});