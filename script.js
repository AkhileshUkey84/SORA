// Application State
let currentUser = null;
let currentDataset = null;
let queryHistory = [];
let isSignUp = false;

// DOM Elements
const authPage = document.getElementById('auth-page');
const dashboardPage = document.getElementById('dashboard-page');
const authForm = document.getElementById('auth-form');
const authTitle = document.getElementById('auth-title');
const authSubtitle = document.getElementById('auth-subtitle');
const authSubmit = document.getElementById('auth-submit');
const toggleAuth = document.getElementById('toggle-auth');
const nameFields = document.getElementById('name-fields');
const passwordHint = document.getElementById('password-hint');

// Navigation
const navItems = document.querySelectorAll('.nav-item');
const contentSections = document.querySelectorAll('.content-section');
const pageTitle = document.getElementById('page-title');
const currentUserEmail = document.getElementById('current-user-email');
const logoutBtn = document.getElementById('logout-btn');

// File Upload
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

// Query Interface
const questionForm = document.getElementById('question-form');
const questionInput = document.getElementById('question-input');
const askBtn = document.getElementById('ask-btn');
const datasetInfo = document.getElementById('dataset-info');
const resultsSection = document.getElementById('results-section');
const suggestionBtns = document.querySelectorAll('.suggestion-btn');

// History
const searchQueries = document.getElementById('search-queries');
const filterBtns = document.querySelectorAll('.filter-btn');
const historyList = document.getElementById('history-list');
const totalQueries = document.getElementById('total-queries');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    // Check if user is logged in (demo purposes)
    const savedUser = localStorage.getItem('currentUser');
    if (savedUser) {
        currentUser = JSON.parse(savedUser);
        showDashboard();
    } else {
        showAuth();
    }
    
    // Load saved data
    loadSavedData();
}

function setupEventListeners() {
    // Auth form
    authForm.addEventListener('submit', handleAuthSubmit);
    toggleAuth.addEventListener('click', toggleAuthMode);
    
    // Navigation
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const section = item.dataset.section;
            switchSection(section);
        });
    });
    
    // Logout
    logoutBtn.addEventListener('click', handleLogout);
    
    // File upload
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', handleDragOver);
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Query form
    questionForm.addEventListener('submit', handleQuestionSubmit);
    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            questionInput.value = btn.textContent;
        });
    });
    
    // History search and filters
    searchQueries.addEventListener('input', filterHistory);
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            filterHistory();
        });
    });
}

// Authentication
function toggleAuthMode() {
    isSignUp = !isSignUp;
    
    if (isSignUp) {
        authTitle.textContent = 'Create Account';
        authSubtitle.textContent = 'Sign up to start analyzing your data with AI';
        authSubmit.textContent = 'Create Account';
        toggleAuth.textContent = 'Already have an account? Sign in';
        nameFields.style.display = 'flex';
        passwordHint.style.display = 'block';
        toggleAuth.classList.remove('toggle-auth');
        toggleAuth.classList.add('toggle-auth');
        toggleAuth.style.border = '2px solid #e5e7eb';
        toggleAuth.style.background = 'white';
    } else {
        authTitle.textContent = 'Welcome Back';
        authSubtitle.textContent = 'Sign in to your AI Data Analyst account';
        authSubmit.textContent = 'Sign In';
        toggleAuth.textContent = 'Don\'t have an account? Sign up';
        nameFields.style.display = 'none';
        passwordHint.style.display = 'none';
        toggleAuth.classList.add('toggle-auth');
        toggleAuth.style.border = 'none';
        toggleAuth.style.background = 'none';
    }
}

function handleAuthSubmit(e) {
    e.preventDefault();
    
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const firstName = document.getElementById('firstName').value;
    const lastName = document.getElementById('lastName').value;
    
    // Simple validation
    if (!email || !password) {
        showToast('Please fill in all required fields', 'error');
        return;
    }
    
    if (password.length < 6) {
        showToast('Password must be at least 6 characters', 'error');
        return;
    }
    
    // Simulate API call
    authSubmit.disabled = true;
    authSubmit.innerHTML = `
        <div style="display: flex; align-items: center;">
            <div style="border: 2px solid transparent; border-top: 2px solid white; border-radius: 50%; width: 16px; height: 16px; animation: spin 1s linear infinite; margin-right: 8px;"></div>
            ${isSignUp ? 'Creating Account...' : 'Signing In...'}
        </div>
    `;
    
    setTimeout(() => {
        currentUser = {
            email,
            firstName: firstName || email.split('@')[0],
            lastName: lastName || '',
        };
        
        localStorage.setItem('currentUser', JSON.stringify(currentUser));
        
        showToast(isSignUp ? 'Account created successfully!' : 'Welcome back!', 'success');
        showDashboard();
        
        authSubmit.disabled = false;
        authSubmit.textContent = isSignUp ? 'Create Account' : 'Sign In';
    }, 1500);
}

function handleLogout() {
    currentUser = null;
    currentDataset = null;
    queryHistory = [];
    localStorage.removeItem('currentUser');
    localStorage.removeItem('currentDataset');
    localStorage.removeItem('queryHistory');
    
    showToast('Logged out successfully', 'success');
    showAuth();
}

// Page Management
function showAuth() {
    authPage.classList.add('active');
    dashboardPage.classList.remove('active');
}

function showDashboard() {
    authPage.classList.remove('active');
    dashboardPage.classList.add('active');
    
    currentUserEmail.textContent = currentUser.email;
    updateDatasetInfo();
    updateHistory();
}

function switchSection(section) {
    // Update navigation
    navItems.forEach(item => {
        item.classList.toggle('active', item.dataset.section === section);
    });
    
    // Update content
    contentSections.forEach(section_el => {
        section_el.classList.remove('active');
    });
    
    const targetSection = document.getElementById(`${section}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }
    
    // Update page title
    const titles = {
        upload: 'Dashboard',
        query: 'Ask Question',
        history: 'Query History'
    };
    pageTitle.textContent = titles[section] || 'Dashboard';
}

// File Upload
function handleDragOver(e) {
    e.preventDefault();
    dropzone.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    
    if (file.size > 100 * 1024 * 1024) {
        showToast('File size must be less than 100MB', 'error');
        return;
    }
    
    // Show progress
    progressSection.style.display = 'block';
    let progress = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress > 100) progress = 100;
        
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Uploading... ${Math.floor(progress)}%`;
        
        if (progress >= 100) {
            clearInterval(interval);
            
            // Simulate file processing
            setTimeout(() => {
                finishUpload(file);
            }, 500);
        }
    }, 200);
}

function finishUpload(file) {
    // Create mock dataset
    currentDataset = {
        id: Date.now().toString(),
        name: file.name.replace('.csv', ''),
        originalName: file.name,
        rowCount: Math.floor(Math.random() * 10000) + 100,
        columnCount: Math.floor(Math.random() * 15) + 5,
        uploadDate: new Date().toISOString(),
        columns: generateMockColumns(),
        sampleData: generateMockData()
    };
    
    localStorage.setItem('currentDataset', JSON.stringify(currentDataset));
    
    progressSection.style.display = 'none';
    progressFill.style.width = '0%';
    
    showToast(`Dataset "${file.name}" uploaded successfully!`, 'success');
    
    // Switch to query section
    switchSection('query');
    updateDatasetInfo();
}

function generateMockColumns() {
    const possibleColumns = [
        'id', 'name', 'email', 'age', 'salary', 'department', 'city', 'country',
        'product_name', 'price', 'quantity', 'category', 'date', 'revenue',
        'customer_id', 'order_id', 'status', 'rating', 'description'
    ];
    
    const numColumns = Math.floor(Math.random() * 10) + 5;
    return possibleColumns.slice(0, numColumns);
}

function generateMockData() {
    if (!currentDataset) return [];
    
    const data = [];
    for (let i = 0; i < 5; i++) {
        const row = {};
        currentDataset.columns.forEach(col => {
            if (col === 'id') {
                row[col] = i + 1;
            } else if (col.includes('name')) {
                row[col] = `Sample ${col} ${i + 1}`;
            } else if (col.includes('email')) {
                row[col] = `user${i + 1}@example.com`;
            } else if (col.includes('age')) {
                row[col] = Math.floor(Math.random() * 50) + 20;
            } else if (col.includes('price') || col.includes('salary') || col.includes('revenue')) {
                row[col] = Math.floor(Math.random() * 100000) + 1000;
            } else if (col.includes('date')) {
                row[col] = new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            } else {
                row[col] = `Value ${i + 1}`;
            }
        });
        data.push(row);
    }
    return data;
}

// Query Interface
function updateDatasetInfo() {
    if (currentDataset) {
        datasetInfo.style.display = 'flex';
        document.getElementById('dataset-name').textContent = currentDataset.originalName;
        document.getElementById('dataset-rows').textContent = currentDataset.rowCount.toLocaleString();
        document.getElementById('dataset-columns').textContent = currentDataset.columnCount;
        document.getElementById('dataset-date').textContent = formatDate(currentDataset.uploadDate);
    } else {
        datasetInfo.style.display = 'none';
    }
}

function handleQuestionSubmit(e) {
    e.preventDefault();
    
    const question = questionInput.value.trim();
    if (!question) {
        showToast('Please enter a question about your data', 'error');
        return;
    }
    
    if (!currentDataset) {
        showToast('Please upload a dataset first', 'error');
        switchSection('upload');
        return;
    }
    
    // Show loading state
    askBtn.disabled = true;
    askBtn.innerHTML = `
        <div style="display: flex; align-items: center;">
            <div style="border: 2px solid transparent; border-top: 2px solid white; border-radius: 50%; width: 16px; height: 16px; animation: spin 1s linear infinite; margin-right: 8px;"></div>
            Processing...
        </div>
    `;
    
    // Simulate AI processing
    setTimeout(() => {
        processQuery(question);
        askBtn.disabled = false;
        askBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22,2 15,22 11,13 2,9"/>
            </svg>
            Ask
        `;
    }, 2000);
}

function processQuery(question) {
    // Generate mock SQL and results
    const mockSQL = generateMockSQL(question);
    const mockResults = generateMockResults(question);
    const explanation = generateMockExplanation(question);
    
    // Create query record
    const query = {
        id: Date.now().toString(),
        question,
        sql: mockSQL,
        results: mockResults,
        explanation,
        status: 'success',
        timestamp: new Date().toISOString(),
        datasetName: currentDataset.name
    };
    
    queryHistory.unshift(query);
    localStorage.setItem('queryHistory', JSON.stringify(queryHistory));
    
    // Show results
    showQueryResults(query);
    updateHistory();
    
    showToast(`Found ${mockResults.length} results`, 'success');
}

function generateMockSQL(question) {
    const tableName = currentDataset.name.toLowerCase().replace(/[^a-z0-9]/g, '_');
    
    if (question.toLowerCase().includes('top')) {
        return `SELECT * FROM ${tableName} ORDER BY ${currentDataset.columns[1]} DESC LIMIT 10;`;
    } else if (question.toLowerCase().includes('count') || question.toLowerCase().includes('total')) {
        return `SELECT COUNT(*) as total FROM ${tableName};`;
    } else if (question.toLowerCase().includes('average') || question.toLowerCase().includes('avg')) {
        return `SELECT AVG(${currentDataset.columns[2]}) as average FROM ${tableName};`;
    } else {
        return `SELECT * FROM ${tableName} LIMIT 20;`;
    }
}

function generateMockResults(question) {
    if (question.toLowerCase().includes('count') || question.toLowerCase().includes('total')) {
        return [{ total: Math.floor(Math.random() * 1000) + 100 }];
    } else if (question.toLowerCase().includes('average') || question.toLowerCase().includes('avg')) {
        return [{ average: (Math.random() * 100).toFixed(2) }];
    } else {
        return currentDataset.sampleData.slice(0, Math.floor(Math.random() * 5) + 3);
    }
}

function generateMockExplanation(question) {
    const explanations = [
        "This query retrieves the data you requested by filtering and sorting the records based on your criteria.",
        "I've analyzed your dataset and generated a SQL query to answer your question about the data patterns.",
        "The query examines your data and provides insights based on the specific question you asked.",
        "This analysis looks at your dataset to find the information you're looking for using SQL operations."
    ];
    
    return explanations[Math.floor(Math.random() * explanations.length)];
}

function showQueryResults(query) {
    resultsSection.style.display = 'block';
    
    // Update status badge
    const statusBadge = document.getElementById('status-badge');
    statusBadge.textContent = query.status;
    statusBadge.className = `status-badge ${query.status}`;
    
    // Show explanation
    const aiExplanation = document.getElementById('ai-explanation');
    const explanationText = document.getElementById('explanation-text');
    if (query.explanation) {
        aiExplanation.style.display = 'block';
        explanationText.textContent = query.explanation;
    } else {
        aiExplanation.style.display = 'none';
    }
    
    // Show SQL
    const sqlSection = document.getElementById('sql-section');
    const sqlCode = document.getElementById('sql-code');
    if (query.sql) {
        sqlSection.style.display = 'block';
        sqlCode.textContent = query.sql;
    } else {
        sqlSection.style.display = 'none';
    }
    
    // Show results table
    const resultsTable = document.getElementById('results-table');
    if (query.results && query.results.length > 0) {
        resultsTable.innerHTML = createResultsTable(query.results);
    } else {
        resultsTable.innerHTML = '<p>No results found.</p>';
    }
}

function createResultsTable(results) {
    if (!results || results.length === 0) return '<p>No results found.</p>';
    
    const columns = Object.keys(results[0]);
    
    let html = '<table><thead><tr>';
    columns.forEach(col => {
        html += `<th>${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    results.forEach(row => {
        html += '<tr>';
        columns.forEach(col => {
            const value = row[col];
            const displayValue = typeof value === 'number' ? value.toLocaleString() : String(value);
            html += `<td>${displayValue}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    if (results.length > 20) {
        html += `<p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 1rem;">Showing first 20 of ${results.length} results</p>`;
    }
    
    return html;
}

// History Management
function updateHistory() {
    totalQueries.textContent = queryHistory.length;
    renderHistory();
}

function filterHistory() {
    const searchTerm = searchQueries.value.toLowerCase();
    const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
    
    let filtered = queryHistory;
    
    if (searchTerm) {
        filtered = filtered.filter(query => 
            query.question.toLowerCase().includes(searchTerm)
        );
    }
    
    if (activeFilter !== 'all') {
        filtered = filtered.filter(query => query.status === activeFilter);
    }
    
    renderHistory(filtered);
}

function renderHistory(queries = queryHistory) {
    if (queries.length === 0) {
        historyList.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                    <path d="M3 3v5h5"/>
                    <path d="M12 7v5l4 2"/>
                </svg>
                <h3>${queryHistory.length === 0 ? 'No query history yet' : 'No queries match your search'}</h3>
                <p>${queryHistory.length === 0 ? 'Start by uploading a dataset and asking your first question' : 'Try adjusting your search terms or filters'}</p>
                ${queryHistory.length === 0 ? '<button class="upload-btn" onclick="switchSection(\'upload\')">Upload Dataset</button>' : ''}
            </div>
        `;
        return;
    }
    
    let html = '';
    queries.forEach(query => {
        html += createHistoryItem(query);
    });
    
    historyList.innerHTML = html;
}

function createHistoryItem(query) {
    const statusIcon = query.status === 'success' 
        ? '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22,4 12,14.01 9,11.01"/></svg>'
        : '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4"/><path d="M12 17h.01"/><path d="M12 3c-4.97 0-9 4.03-9 9s4.03 9 9 9 9-4.03 9-9-4.03-9-9-9z"/></svg>';
    
    return `
        <div class="history-item">
            <div class="history-item-header">
                <h4>${query.question}</h4>
                <div class="history-meta">
                    <span>${formatRelativeTime(query.timestamp)}</span>
                    <span class="status-badge ${query.status}">
                        ${statusIcon}
                        <span style="margin-left: 4px; text-transform: capitalize;">${query.status}</span>
                    </span>
                </div>
            </div>
            
            ${query.sql ? `
                <div class="history-sql">
                    <p>Generated SQL:</p>
                    <code>${query.sql}</code>
                </div>
            ` : ''}
            
            <div class="history-footer">
                <span>
                    Dataset: ${query.datasetName || 'Unknown'}
                    ${query.results ? ` • ${query.results.length} results` : ''}
                </span>
                <div class="history-actions">
                    ${query.status === 'success' ? '<button>View Results</button>' : ''}
                    <button onclick="replayQuery('${query.question}')">Replay Query</button>
                </div>
            </div>
        </div>
    `;
}

function replayQuery(question) {
    switchSection('query');
    questionInput.value = question;
    questionInput.focus();
}

// Utility Functions
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatRelativeTime(dateString) {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffHours < 1) return 'Just now';
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
    });
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-weight: 500;
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

function loadSavedData() {
    // Load saved dataset
    const savedDataset = localStorage.getItem('currentDataset');
    if (savedDataset) {
        currentDataset = JSON.parse(savedDataset);
    }
    
    // Load saved queries
    const savedQueries = localStorage.getItem('queryHistory');
    if (savedQueries) {
        queryHistory = JSON.parse(savedQueries);
    }
}

// Global function for navigation (used in HTML)
window.switchSection = switchSection;

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
`;
document.head.appendChild(style);