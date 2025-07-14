import { createAIOrchestrator, setApiKey } from "./ai-lib/index.js";
setApiKey("groq", "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");

// Direct all optimizer LLM calls to your backend’s /chat/ endpoint:
const ai = createAIOrchestrator({
  baseURL: "http://127.0.0.1:8000",
  endpoints: {
    llm:  "/chat/",
    chat: "/chat/"
  }
});
console.log("AI Orchestrator endpoints:", ai);
console.log("📘 script.js loaded");
let sopUploaded = false;
let datasetUploaded = false;
let currentAnalysis = { violations: [] };
let datasetRowCount = 0

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
      datasetRowCount = result.dataset_info.rows;
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
  document.getElementById('totalViolations').textContent     = analysis.summary.total_violations;
  document.getElementById('complianceRate').textContent      = `${analysis.summary.compliance_rate.toFixed(1)}%`;
  document.getElementById('criticalViolations').textContent  = analysis.summary.critical;
  document.getElementById('compliantRows').textContent       = analysis.summary.compliant_rows;

  // Create the pie/donut charts
  createSeverityChart(analysis.summary);
  createComplianceChart(analysis.summary);

  // Build a numeric array of all violation row numbers
  const violationRows = analysis.violations.map(v => Number(v.row_number));
  // If there are no violations, fall back to 1 so the chart still renders
  const maxViolationRow = violationRows.length ? Math.max(...violationRows) : 1;

  // Draw the timeline chart from x=1…maxViolationRow
  createTimelineChart(analysis.violations, maxViolationRow);

  // Finally, render the detailed violation cards
  displayViolations(analysis.violations);

  // Show the results section
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

// ─── Replace your existing toggleViolationDetails with this ───

// Replace your existing toggleViolationDetails function with this:
async function toggleViolationDetails(index, buttonElement) {
  const detailsContainer = document.getElementById(`details-container-${index}`);
  const detailsContent   = document.getElementById(`details-content-${index}`);
  const isVisible        = detailsContainer.style.display !== 'none';

  if (isVisible) {
    detailsContainer.style.display = 'none';
    buttonElement.innerHTML        = '<i class="fas fa-eye me-1"></i>Details';
    buttonElement.classList.replace('btn-outline-secondary','btn-outline-primary');
  } else {
    detailsContainer.style.display = 'block';
    buttonElement.innerHTML        = '<i class="fas fa-eye-slash me-1"></i>Hide Details';
    buttonElement.classList.replace('btn-outline-primary','btn-outline-secondary');

    if (detailsContent.innerHTML.includes('Loading detailed analysis')) {
      try {
        const res = await fetch('http://127.0.0.1:8000/violation-details/', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({violation_index: index})
        });
        const { success, detailed_explanation: explanation, error } = await res.json();

        if (!success) {
          detailsContent.innerHTML = `
            <div class="alert alert-warning">
              <i class="fas fa-exclamation-triangle me-2"></i>
              Unable to load detailed analysis: ${error||'Unknown'}
            </div>`;
          return;
        }

        // Debug the explanation structure
        debugExplanationStructure(explanation);

        // Format markdown/text
        const formattedText = formatDetailedExplanation(explanation);

        // Create four donut-chart containers + the text block
        detailsContent.innerHTML = `
          <div class="violation-charts" 
               style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
            <div><h6>Root Cause Analysis</h6><div id="donut-RCA-${index}" style="height:200px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"></div></div>
            <div><h6>Business Impact</h6><div id="donut-BI-${index}"  style="height:200px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"></div></div>
            <div><h6>Technical Details</h6><div id="donut-TD-${index}"  style="height:200px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"></div></div>
            <div><h6>Corrective Actions</h6><div id="donut-CA-${index}"  style="height:200px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;"></div></div>
          </div>
          <div class="details-text" id="text-details-${index}">${formattedText}</div>
        `;

        // Add a longer delay and verify DOM elements exist
        setTimeout(() => {
          console.log("About to render section donuts...");
          const elements = ['RCA', 'BI', 'TD', 'CA'].map(id => document.getElementById(`donut-${id}-${index}`));
          console.log("DOM elements found:", elements.map(el => el ? 'YES' : 'NO'));
          
          renderSectionDonuts(index, explanation);
        }, 200);

      } catch (err) {
        console.error('Error in toggleViolationDetails:', err);
        detailsContent.innerHTML = `
          <div class="alert alert-danger">
            <i class="fas fa-times-circle me-2"></i>
            Error loading detailed analysis. Please try again.
          </div>`;
      }
    }
  }
}

function renderViolationDetailsCharts(index, explanation) {
  // Define the sections we expect
  const sections = [
    "Root Cause Analysis",
    "Business Impact",
    "Technical Details",
    "Step-by-Step Remediation",
    "Prevention Measures"
  ];

  // Count bullets in each
  const counts = sections.map(title => {
    // Grab the text for this section
    const re = new RegExp(title + "[\\s\\S]*?(?=(?:[A-Z][a-z ]+:)|$)");
    const match = explanation.match(re);
    if (!match) return 0;
    // Count lines that start with "- "
    return match[0]
      .split("\n")
      .filter(line => line.trim().startsWith("-"))
      .length;
  });

  // Build Plotly bar chart
  const data = [{
    x: sections,
    y: counts,
    type: 'bar',
    marker: { color: '#364FC7' }
  }];

  const layout = {
    title: 'Detail Sections Breakdown',
    margin: { t: 40, l: 40, r: 20, b: 80 },
    xaxis: { tickangle: -45 }
  };

  Plotly.newPlot(
    `chart-details-${index}`,
    data,
    layout,
    { responsive: true, displayModeBar: false }
  );
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

/*Chat Sidebar Functionality*/

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
  const rawQuery = chatInput.value.trim();
  if (!rawQuery) return;
  
  // Append user message
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-msg chat-msg-user mb-2';
  userDiv.textContent = rawQuery;
  chatMessages.appendChild(userDiv);
  chatInput.value = '';
  chatMessages.scrollTop = chatMessages.scrollHeight;
  
  // Process query through AI orchestrator
  try {
  const chatHistory = Array.from(chatMessages.querySelectorAll('.chat-msg')).map(msg => ({
    role: msg.classList.contains('chat-msg-user') ? 'user' : 'assistant',
    content: msg.textContent
  }));

  const { response: optimizedQuery } = await ai.processQuery(rawQuery, chatHistory);
  console.log("Optimized query is:", optimizedQuery);

  // Fallback if optimizer returns empty or only a classification label
  const classifierKeywords = [
    "LLM_Summary",
    "LLM_Creation",
    "LLM_Ideation",
    "LLM_Analysis",
    "LLM_Converter",
    "LLM_Default"
  ];
  const finalQuery = (!optimizedQuery || classifierKeywords.includes(optimizedQuery.trim()))
    ? rawQuery
    : optimizedQuery;

  if (!optimizedQuery || classifierKeywords.includes(optimizedQuery.trim())) {
    console.warn("Optimizer returned empty or only a classification keyword—falling back to raw user query.");
  }

  const res = await fetch('http://127.0.0.1:8000/chat/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: finalQuery })
  });
  const { response } = await res.json();

  const botDiv = document.createElement('div');
  botDiv.className = 'chat-msg chat-msg-bot mb-2';
  botDiv.textContent = response || 'Sorry, no response.';

  chatMessages.appendChild(botDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
 catch (err) {
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

// ─── Append this at the bottom of script.js ───

function renderSectionDonuts(index, explanation) {
  console.log("=== Rendering section donuts for index:", index);
  console.log("=== Full explanation content:");
  console.log(explanation);
  console.log("=== End explanation content");
  
  const sections = [
    { 
      key: "Root Cause Analysis", 
      id: "RCA", 
      patterns: [
        /root\s+cause\s+analysis/i,
        /root\s+cause/i,
        /cause\s+analysis/i,
        /\*\*root\s+cause/i,
        /###\s*root\s+cause/i
      ]
    },
    { 
      key: "Business Impact", 
      id: "BI", 
      patterns: [
        /business\s+impact/i,
        /impact\s+on\s+business/i,
        /\*\*business\s+impact/i,
        /###\s*business\s+impact/i,
        /impact\s+assessment/i
      ]
    },
    { 
      key: "Technical Details", 
      id: "TD", 
      patterns: [
        /technical\s+details/i,
        /technical\s+analysis/i,
        /\*\*technical\s+details/i,
        /###\s*technical\s+details/i,
        /technical\s+information/i
      ]
    },
    { 
      key: "Corrective Actions", 
      id: "CA", 
      patterns: [
        /corrective\s+actions/i,
        /remediation/i,
        /step[-\s]by[-\s]step/i,
        /prevention/i,
        /\*\*corrective/i,
        /###\s*corrective/i,
        /recommended\s+actions/i,
        /action\s+plan/i
      ]
    }
  ];

  sections.forEach(({ key, id, patterns }) => {
    console.log(`\n--- Processing section: ${key} ---`);
    
    let sectionText = null;
    let matchedPattern = null;
    
    // Try each pattern
    for (const pattern of patterns) {
      console.log(`Trying pattern: ${pattern}`);
      
      // Find the section header
      const headerMatch = explanation.match(pattern);
      if (headerMatch) {
        console.log(`Found header match: "${headerMatch[0]}" at position ${headerMatch.index}`);
        
        // Extract text from the header position to the next major section or end
        const startPos = headerMatch.index;
        const textAfterHeader = explanation.substring(startPos);
        
        // Find the end of this section (next major header or end of text)
        const nextSectionMatch = textAfterHeader.match(/\n\n(?:[A-Z][a-z\s]+:|###|##|\*\*[A-Z])/);
        const endPos = nextSectionMatch ? nextSectionMatch.index : textAfterHeader.length;
        
        sectionText = textAfterHeader.substring(0, endPos);
        matchedPattern = pattern;
        console.log(`Extracted section text (${sectionText.length} chars):`, sectionText.substring(0, 200) + '...');
        break;
      }
    }
    
    // If no pattern matched, try a more general approach
    if (!sectionText) {
      console.log(`No pattern matched for ${key}, trying general approach...`);
      
      // Split explanation into paragraphs and look for any that contain key terms
      const paragraphs = explanation.split(/\n\n+/);
      const keyTerms = key.toLowerCase().split(' ');
      
      for (const paragraph of paragraphs) {
        const lowerPara = paragraph.toLowerCase();
        if (keyTerms.every(term => lowerPara.includes(term))) {
          sectionText = paragraph;
          console.log(`Found section via general search: ${paragraph.substring(0, 100)}...`);
          break;
        }
      }
    }
    
    // Extract bullets from the section text
    let bulletLines = [];
    
    if (sectionText) {
      console.log(`Processing bullets from section text...`);
      
      // Try multiple bullet extraction methods
      const methods = [
        // Method 1: Lines starting with - or •
        () => sectionText
          .split('\n')
          .filter(line => line.trim().match(/^[-•]\s+/))
          .map(line => line.replace(/^[-•]\s*/, '').trim()),
        
        // Method 2: Lines starting with numbers
        () => sectionText
          .split('\n')
          .filter(line => line.trim().match(/^\d+\.\s+/))
          .map(line => line.replace(/^\d+\.\s*/, '').trim()),
        
        // Method 3: Any line that looks like a list item
        () => sectionText
          .split('\n')
          .filter(line => {
            const trimmed = line.trim();
            return trimmed.length > 10 && 
                   (trimmed.match(/^[-•*]\s+/) || 
                    trimmed.match(/^\d+\.\s+/) ||
                    trimmed.match(/^[A-Z][a-z]+:/));
          })
          .map(line => line.replace(/^[-•*\d\.]\s*/, '').replace(/^[A-Z][a-z]+:\s*/, '').trim()),
        
        // Method 4: Split by sentences and take meaningful ones
        () => sectionText
          .split(/[.!?]\s+/)
          .filter(sentence => sentence.trim().length > 15 && sentence.trim().length < 100)
          .slice(0, 5) // Take up to 5 sentences
      ];
      
      for (const method of methods) {
        try {
          const extracted = method();
          if (extracted.length > 0) {
            bulletLines = extracted;
            console.log(`Method succeeded, found ${bulletLines.length} items:`, bulletLines);
            break;
          }
        } catch (e) {
          console.log(`Method failed:`, e);
        }
      }
    }
    
    // If still no bullets, create some generic ones based on section type
    if (bulletLines.length === 0) {
      console.log(`No bullets found, creating generic ones for ${key}`);
      
      const genericContent = {
        'Root Cause Analysis': ['Data Quality Issue', 'Process Gap', 'System Error'],
        'Business Impact': ['Compliance Risk', 'Operational Impact', 'Cost Implications'],
        'Technical Details': ['Data Validation', 'System Check', 'Field Analysis'],
        'Corrective Actions': ['Fix Data', 'Update Process', 'Implement Controls']
      };
      
      bulletLines = genericContent[key] || ['Analysis Required', 'Review Needed', 'Action Pending'];
    }
    
    // Clean and format the bullet lines
    bulletLines = bulletLines
      .map(line => {
        let txt = line.replace(/\*\*/g, '')  // Remove markdown bold
                     .replace(/[:\-–]/g, ' ') // Replace colons and dashes with spaces
                     .trim();
        
        // Take first 2-4 meaningful words
        const words = txt.split(/\s+/).filter(word => word.length > 1);
        txt = words.slice(0, 4).join(' ');
        
        return txt.length > 30 ? txt.slice(0, 27) + '...' : txt;
      })
      .filter(label => label.length > 2)
      .slice(0, 8); // Limit to 8 items max
    
    console.log(`Final bullet lines for ${key}:`, bulletLines);
    
    // Create the pie chart
    const values = bulletLines.map(() => 1);
    const colors = generateColors(bulletLines.length);

    const data = [{
      labels: bulletLines,
      values: values,
      type: 'pie',
      hole: 0.4,
      marker: { colors: colors },
      textinfo: 'label',
      textposition: 'auto',
      hoverinfo: 'label+percent',
      textfont: { size: 9 }
    }];

    const layout = {
      margin: { t: 20, b: 20, l: 20, r: 20 },
      showlegend: false,
      font: { size: 10 }
    };

    try {
      const elementId = `donut-${id}-${index}`;
      console.log(`Creating chart for element: ${elementId}`);
      
      // Check if element exists
      const element = document.getElementById(elementId);
      if (!element) {
        console.error(`Element ${elementId} not found in DOM`);
        return;
      }
      
      Plotly.newPlot(elementId, data, layout, {
        responsive: true, displayModeBar: false
      });
      console.log(`✓ Successfully created donut for ${key}`);
    } catch (error) {
      console.error(`✗ Error creating donut for ${key}:`, error);
    }
  });
}

// Add this helper function to debug the explanation structure:
function debugExplanationStructure(explanation) {
  console.log("=== DEBUGGING EXPLANATION STRUCTURE ===");
  console.log("Total length:", explanation.length);
  console.log("Number of lines:", explanation.split('\n').length);
  console.log("Number of paragraphs:", explanation.split('\n\n').length);
  
  // Show all headers/sections found
  const possibleHeaders = explanation.match(/^.{1,50}:|\*\*.{1,50}\*\*|###.{1,50}|##.{1,50}/gm);
  console.log("Possible headers found:", possibleHeaders);
  
  // Show all bullet-like lines
  const bulletLines = explanation.split('\n').filter(line => 
    line.trim().match(/^[-•*]\s+/) || line.trim().match(/^\d+\.\s+/)
  );
  console.log("Bullet-like lines found:", bulletLines.length);
  bulletLines.forEach((line, i) => console.log(`  ${i+1}: ${line.trim()}`));
  
  console.log("=== END DEBUG ===");
}
function generateColors(count) {
  const baseColors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
    '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA'
  ];
  
  const colors = [];
  for (let i = 0; i < count; i++) {
    colors.push(baseColors[i % baseColors.length]);
  }
  return colors;
}
function createTimelineChart(violations) {
  const ctx = document.getElementById('timelineChart').getContext('2d');
  const severityMap = { CRITICAL:4, HIGH:3, MEDIUM:2, LOW:1 };

  // 1) labels = [1, 9, 10, 11, 12, 13]
  const labels = violations.map(v => v.row_number);

  // 2) data  = [4, 4, 4, 4, 4, 4]  (or whatever map gives)
  const data   = violations.map(v => severityMap[v.severity.toUpperCase()] || 0);

  new Chart(ctx, {
    type: 'line',
    data: { labels, datasets:[{
      label: 'Severity',
      data,
      fill: false,
      borderColor: '#6c757d',
      pointBackgroundColor: data.map(d =>
        d===4 ? '#dc3545' :
        d===3 ? '#fd7e14' :
        d===2 ? '#ffc107' :
        d===1 ? '#28a745' :
                '#6c757d'
      ),
      pointRadius: 6,
      tension: 0.2
    }]},
    options: {
      responsive: true,
      scales: {
        x: { 
          type: 'category',
          title: { display:true, text:'Data Row Number' }
        },
        y: {
          beginAtZero: true,
          min: 0,
          max: 4,
          ticks: {
            stepSize: 1,
            callback: v => ({0:'COMPLIANT',1:'LOW',2:'MEDIUM',3:'HIGH',4:'CRITICAL'})[v]
          }
        }
      },
      plugins: {
        title: { display:true, text:'Timeline of Violations by Severity' },
        legend: { display:false }
      }
    }
  });
}
