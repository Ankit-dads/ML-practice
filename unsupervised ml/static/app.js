// AuraClust Client Application Logic
document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        activeTab: 'tab-dashboard',
        k: 5,
        dataset: {
            columns: [],
            isCustom: false,
            featureIndices: [3, 4],
            featureNames: ['Annual Income (k$)', 'Spending Score (1-100)'],
            stats: {},
            totalRecords: 0
        },
        clusteringResults: null,
        fullData: [],
        filteredData: [],
        table: {
            currentPage: 1,
            pageSize: 10,
            searchQuery: ''
        },
        charts: {
            elbow: null,
            scatter: null
        }
    };

    // DOM Elements
    const elements = {
        navItems: document.querySelectorAll('.nav-item'),
        tabPanels: document.querySelectorAll('.tab-panel'),
        pageTitle: document.getElementById('page-title'),
        pageSubtitle: document.getElementById('page-subtitle'),
        currentDatasetName: document.getElementById('current-dataset-name'),
        btnResetData: document.getElementById('btn-reset-data'),
        
        // Metrics
        metricTotalCustomers: document.getElementById('metric-total-customers'),
        metricAvgX: document.getElementById('metric-avg-x'),
        metricAvgY: document.getElementById('metric-avg-y'),
        metricSilhouette: document.getElementById('metric-silhouette'),
        labelAvgX: document.getElementById('label-avg-x'),
        labelAvgY: document.getElementById('label-avg-y'),
        
        // Slider Control
        clusterRange: document.getElementById('cluster-range'),
        clusterVal: document.getElementById('cluster-val'),
        descFeatureX: document.getElementById('desc-feature-x'),
        descFeatureY: document.getElementById('desc-feature-y'),
        
        // Segment Breakdown
        segmentsContainer: document.getElementById('segments-container'),
        
        // Predictor
        predictionForm: document.getElementById('prediction-form'),
        labelPredictX: document.getElementById('label-predict-x'),
        labelPredictY: document.getElementById('label-predict-y'),
        suffixPredictX: document.getElementById('suffix-predict-x'),
        hintPredictX: document.getElementById('hint-predict-x'),
        hintPredictY: document.getElementById('hint-predict-y'),
        btnPredict: document.getElementById('btn-predict'),
        predictionEmptyState: document.getElementById('prediction-empty-state'),
        predictionResultContent: document.getElementById('prediction-result-content'),
        resultClusterTitle: document.getElementById('result-cluster-title'),
        resLblX: document.getElementById('res-lbl-x'),
        resLblY: document.getElementById('res-lbl-y'),
        resValX: document.getElementById('res-val-x'),
        resValY: document.getElementById('res-val-y'),
        resultArchetypeDesc: document.getElementById('result-archetype-description'),
        resultCampaignStrategy: document.getElementById('result-campaign-strategy'),
        
        // Data Explorer
        uploadZone: document.getElementById('upload-zone'),
        fileInput: document.getElementById('file-input'),
        uploadProgress: document.getElementById('upload-progress'),
        uploadStatus: document.getElementById('upload-status'),
        tableHeaders: document.getElementById('table-headers'),
        tableBody: document.getElementById('table-body'),
        tableSearch: document.getElementById('table-search'),
        btnExportCsv: document.getElementById('btn-export-csv'),
        paginationInfo: document.getElementById('pagination-info'),
        btnPrevPage: document.getElementById('btn-prev-page'),
        btnNextPage: document.getElementById('btn-next-page')
    };

    // Color definitions for clusters (matches CSS variables)
    const clusterColors = [
        '#ff007f', // Pink (0)
        '#9d4edd', // Purple (1)
        '#00f2fe', // Cyan (2)
        '#f39c12', // Orange (3)
        '#2ecc71', // Green (4)
        '#e74c3c', // Soft red (5)
        '#3498db', // Soft blue (6)
        '#9b59b6', // Soft purple (7)
        '#f1c40f', // Soft yellow (8)
        '#1abc9c'  // Soft teal (9)
    ];

    // Card layouts for clusters (archetypes)
    const archetypeTemplates = {
        "Low Income, High Spend (Careless)": {
            desc: "Low income but high spending habits. These customers prioritize fashion, leisure, and trends over financial caution. Frequently responsive to promotional impulses and aesthetic-focused marketing.",
            campaign: "Direct targeted social media ads, flashy flash sales, and trendy visual recommendations. Offer budget-buy aesthetic-driven highlights.",
            class: "sc-careless",
            pillClass: "sc-careless-pill"
        },
        "High Income, High Spend (Target)": {
            desc: "High income and high spending behavior. This is the premium cluster representing customers who value premium quality, VIP service, exclusivity, and brand status. Highly profitable segment.",
            campaign: "Provide premium VIP customer status, luxury item updates, invite-only event notifications, personal shopper experiences, and premium loyalty perks.",
            class: "sc-target",
            pillClass: "sc-target-pill"
        },
        "High Income, Low Spend (Conservative)": {
            desc: "High income but highly cautious spending habits. These customers are affluent but extremely value-conscious. They perform extensive comparisons and are highly immune to cheap impulse campaigns.",
            campaign: "Provide data-driven newsletters highlighting product durability, high returns on investment, quality warranties, and premium utility packages.",
            class: "sc-conservative",
            pillClass: "sc-conservative-pill"
        },
        "Low Income, Low Spend (Frugal)": {
            desc: "Low income and very low spending habits. Highly price-sensitive. These shoppers make practical, necessity-only purchases and are heavily influenced by heavy discounts and cost-savings.",
            campaign: "Send massive clearance deal emails, BOGO (Buy One Get One) notifications, free shipping codes, and value coupons.",
            class: "sc-frugal",
            pillClass: "sc-frugal-pill"
        },
        "Medium Income, Medium Spend (Standard)": {
            desc: "Moderate income and moderate spending habits. The standard average customer group. They buy reliable goods, respond nicely to general campaigns, and appreciate practical rewards.",
            campaign: "Standard email campaigns, cashback opportunities, holiday discounts, and loyalty club point accelerators.",
            class: "sc-standard",
            pillClass: "sc-standard-pill"
        }
    };

    // Initialize Application
    async function init() {
        setupEventListeners();
        await fetchMetadata();
        await loadElbowCurve();
        await trainClusteringModel();
    }

    // Setup UI Interactions
    function setupEventListeners() {
        // Tab Navigation click handlers
        elements.navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const targetTab = item.getAttribute('data-tab');
                switchTab(targetTab);
            });
        });

        // Cluster count slider handler
        elements.clusterRange.addEventListener('input', (e) => {
            state.k = parseInt(e.target.value);
            elements.clusterVal.textContent = state.k;
        });

        elements.clusterRange.addEventListener('change', async () => {
            await trainClusteringModel();
        });

        // Prediction form handler
        elements.predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await performPrediction();
        });

        // Reset dataset button
        elements.btnResetData.addEventListener('click', async () => {
            await resetDataset();
        });

        // Table search bar
        elements.tableSearch.addEventListener('input', (e) => {
            state.table.searchQuery = e.target.value;
            state.table.currentPage = 1;
            applyTableFilters();
            renderTableBody();
        });

        // Export segmented CSV
        elements.btnExportCsv.addEventListener('click', () => {
            exportToCSV();
        });

        // Pagination buttons
        elements.btnPrevPage.addEventListener('click', () => {
            if (state.table.currentPage > 1) {
                state.table.currentPage--;
                renderTableBody();
            }
        });

        elements.btnNextPage.addEventListener('click', () => {
            const maxPage = Math.ceil(state.filteredData.length / state.table.pageSize);
            if (state.table.currentPage < maxPage) {
                state.table.currentPage++;
                renderTableBody();
            }
        });

        // Custom File Drag and Drop zone listeners
        elements.uploadZone.addEventListener('click', () => elements.fileInput.click());
        elements.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            elements.uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                elements.uploadZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            elements.uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                elements.uploadZone.classList.remove('dragover');
            }, false);
        });

        elements.uploadZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
    }

    // Switch active dashboard tabs
    function switchTab(tabId) {
        state.activeTab = tabId;
        
        // Toggle Sidebar Nav States
        elements.navItems.forEach(item => {
            if (item.getAttribute('data-tab') === tabId) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });

        // Toggle Content Sections
        elements.tabPanels.forEach(panel => {
            if (panel.id === tabId) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });

        // Update titles based on Tab Selection
        let title = "Dashboard Overview";
        let subtitle = "Perform K-Means clustering and review dataset metrics";

        if (tabId === 'tab-analyzer') {
            title = "Cluster Analyzer";
            subtitle = "Visualize clustering results in 2D and adjust engine cluster density";
            // Redraw scatter chart to fix sizing issues inside display-none templates
            if (state.charts.scatter) {
                setTimeout(() => state.charts.scatter.resize(), 50);
            }
        } else if (tabId === 'tab-segments') {
            title = "Segment Breakdown";
            subtitle = "In-depth review of customer personas and target campaigns";
        } else if (tabId === 'tab-predictor') {
            title = "Prospect Predictor";
            subtitle = "Predict cluster classification and marketing campaign for new prospects";
        } else if (tabId === 'tab-explorer') {
            title = "Data Explorer";
            subtitle = "Examine raw records, export segments, or import custom datasets";
        }

        elements.pageTitle.textContent = title;
        elements.pageSubtitle.textContent = subtitle;
    }

    // Fetch active dataset structure and metadata metrics
    async function fetchMetadata() {
        try {
            const res = await fetch('/api/data');
            const data = await res.json();
            
            state.dataset.columns = data.columns;
            state.dataset.isCustom = data.is_custom;
            state.dataset.featureIndices = data.feature_indices;
            state.dataset.featureNames = data.feature_names;
            state.dataset.stats = data.stats;
            state.dataset.totalRecords = data.data_count;
            
            // Adjust UI based on defaults vs uploaded files
            if (state.dataset.isCustom) {
                elements.currentDatasetName.textContent = "Custom: Uploaded_Dataset.csv";
                elements.btnResetData.style.display = "inline-flex";
            } else {
                elements.currentDatasetName.textContent = "Default: Mall_Customers.csv";
                elements.btnResetData.style.display = "none";
            }

            // Update details texts
            elements.descFeatureX.textContent = state.dataset.featureNames[0];
            elements.descFeatureY.textContent = state.dataset.featureNames[1];
            
            // Adjust Predictor form labels
            elements.labelPredictX.textContent = state.dataset.featureNames[0];
            elements.labelPredictY.textContent = state.dataset.featureNames[1];
            elements.resLblX.textContent = state.dataset.featureNames[0];
            elements.resLblY.textContent = state.dataset.featureNames[1];
            
            // Clean suffixes
            if (state.dataset.featureNames[0].toLowerCase().includes('income')) {
                elements.suffixPredictX.textContent = 'k$';
                elements.hintPredictX.textContent = `Range: ${Math.round(state.dataset.stats[state.dataset.featureNames[0]]?.min || 0)} - ${Math.round(state.dataset.stats[state.dataset.featureNames[0]]?.max || 0)}`;
            } else {
                elements.suffixPredictX.textContent = '';
                elements.hintPredictX.textContent = `Range: ${Math.round(state.dataset.stats[state.dataset.featureNames[0]]?.min || 0)} - ${Math.round(state.dataset.stats[state.dataset.featureNames[0]]?.max || 0)}`;
            }
            
            elements.hintPredictY.textContent = `Range: ${Math.round(state.dataset.stats[state.dataset.featureNames[1]]?.min || 0)} - ${Math.round(state.dataset.stats[state.dataset.featureNames[1]]?.max || 0)}`;

            // Populate form default inputs dynamically to fit dataset ranges
            const f1_mean = state.dataset.stats[state.dataset.featureNames[0]]?.mean || 50;
            const f2_mean = state.dataset.stats[state.dataset.featureNames[1]]?.mean || 50;
            document.getElementById('predict-x').value = Math.round(f1_mean);
            document.getElementById('predict-y').value = Math.round(f2_mean);

        } catch (err) {
            console.error("Error fetching dataset metadata:", err);
        }
    }

    // Load WCSS Elbow Data and Render Plotly/Chartjs Curve
    async function loadElbowCurve() {
        try {
            const res = await fetch('/api/elbow');
            const result = await res.json();
            
            if (result.status === 'success') {
                const elbowData = result.data;
                renderElbowChart(elbowData.labels, elbowData.wcss);
            }
        } catch (err) {
            console.error("Error loading elbow optimization data:", err);
        }
    }

    // Train the active K-Means clustering model
    async function trainClusteringModel() {
        try {
            // Setup loading status
            elements.metricSilhouette.textContent = "Loading...";
            
            const res = await fetch(`/api/train?k=${state.k}`);
            const result = await res.json();
            
            if (result.status === 'success') {
                state.clusteringResults = result.data;
                state.fullData = result.data.clean_df_json;
                
                // Update general overview metrics
                elements.metricTotalCustomers.textContent = state.fullData.length;
                elements.metricSilhouette.textContent = state.clusteringResults.silhouette_score.toFixed(4);
                
                const f1 = state.dataset.featureNames[0];
                const f2 = state.dataset.featureNames[1];
                const avg_x = state.dataset.stats[f1]?.mean || 0;
                const avg_y = state.dataset.stats[f2]?.mean || 0;

                // Format metric display
                if (f1.toLowerCase().includes('income')) {
                    elements.metricAvgX.textContent = `$${avg_x.toFixed(1)}k`;
                } else {
                    elements.metricAvgX.textContent = avg_x.toFixed(1);
                }
                
                if (f2.toLowerCase().includes('score')) {
                    elements.metricAvgY.textContent = `${avg_y.toFixed(1)}/100`;
                } else {
                    elements.metricAvgY.textContent = avg_y.toFixed(1);
                }

                // Render dynamic charts and UI cards
                renderScatterChart();
                renderSegmentBreakdown();
                
                // Setup and load Data Explorer table
                applyTableFilters();
                renderTableHeaders();
                renderTableBody();
            }
        } catch (err) {
            console.error("Error training clustering model:", err);
            elements.metricSilhouette.textContent = "Error";
        }
    }

    // Perform customer classification request
    async function performPrediction() {
        try {
            elements.btnPredict.disabled = true;
            elements.btnPredict.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
            
            const x_val = parseFloat(document.getElementById('predict-x').value);
            const y_val = parseFloat(document.getElementById('predict-y').value);
            
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: x_val, y: y_val })
            });
            const result = await res.json();
            
            if (result.status === 'success') {
                const clusterId = result.prediction;
                
                // Find cluster definition
                const clusterInfo = state.clusteringResults.cluster_info.find(c => c.id === clusterId);
                const shortName = clusterInfo ? clusterInfo.short_name : "Standard";
                const template = archetypeTemplates[shortName] || archetypeTemplates["Medium Income, Medium Spend (Standard)"];
                
                // Update prediction result card
                elements.resultClusterTitle.textContent = `Cluster ${clusterId}: ${shortName}`;
                elements.resultClusterTitle.className = ""; // Reset class list
                
                // Format results values
                elements.resValX.textContent = state.dataset.featureNames[0].toLowerCase().includes('income') ? `$${x_val}k` : x_val;
                elements.resValY.textContent = state.dataset.featureNames[1].toLowerCase().includes('score') ? `${y_val}/100` : y_val;
                
                elements.resultArchetypeDesc.textContent = template.desc;
                elements.resultCampaignStrategy.textContent = template.campaign;
                
                // Style prediction box based on color archetype
                const campaignBlock = document.querySelector('.campaign-card');
                campaignBlock.style.borderLeftColor = clusterColors[clusterId % clusterColors.length];
                
                // Reveal panel
                elements.predictionEmptyState.style.display = 'none';
                elements.predictionResultContent.style.display = 'flex';
                
                // Scroll page slightly to focus result on mobile/small viewports
                elements.predictionResultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        } catch (err) {
            console.error("Error running prospect classification prediction:", err);
        } finally {
            elements.btnPredict.disabled = false;
            elements.btnPredict.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Classify Customer';
        }
    }

    // Process drag & drop custom file imports
    function handleFileUpload(file) {
        elements.uploadStatus.className = "upload-status";
        elements.uploadStatus.textContent = "";
        elements.uploadProgress.style.display = "block";
        
        let progress = 0;
        const progressFill = elements.uploadProgress.querySelector('.progress-fill');
        progressFill.style.width = "0%";
        
        // Fake progress animation for aesthetic loading feel
        const progressInterval = setInterval(() => {
            progress += 10;
            progressFill.style.width = `${progress}%`;
            if (progress >= 90) clearInterval(progressInterval);
        }, 100);

        const formData = new FormData();
        formData.append('file', file);

        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(res => res.json())
        .then(async result => {
            clearInterval(progressInterval);
            progressFill.style.width = "100%";
            
            setTimeout(async () => {
                elements.uploadProgress.style.display = "none";
                if (result.status === 'success') {
                    elements.uploadStatus.className = "upload-status success";
                    elements.uploadStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${result.message}`;
                    
                    // Reset explorer state and reload everything
                    state.table.currentPage = 1;
                    state.table.searchQuery = '';
                    elements.tableSearch.value = '';
                    
                    await fetchMetadata();
                    await loadElbowCurve();
                    await trainClusteringModel();
                } else {
                    elements.uploadStatus.className = "upload-status error";
                    elements.uploadStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> ${result.message}`;
                }
            }, 300);
        })
        .catch(err => {
            clearInterval(progressInterval);
            elements.uploadProgress.style.display = "none";
            elements.uploadStatus.className = "upload-status error";
            elements.uploadStatus.innerHTML = `<i class="fa-solid fa-circle-xmark"></i> Connection error. Upload failed.`;
            console.error("File upload error:", err);
        });
    }

    // Reset dataset back to raw Mall Customer details
    async function resetDataset() {
        try {
            const res = await fetch('/api/reset');
            const result = await res.json();
            if (result.status === 'success') {
                elements.uploadStatus.className = "upload-status";
                elements.uploadStatus.textContent = "";
                
                state.table.currentPage = 1;
                state.table.searchQuery = '';
                elements.tableSearch.value = '';
                
                await fetchMetadata();
                await loadElbowCurve();
                await trainClusteringModel();
            }
        } catch (err) {
            console.error("Error resetting dataset:", err);
        }
    }

    // Export clustering output data to client downloadable CSV
    function exportToCSV() {
        if (!state.fullData || state.fullData.length === 0) return;
        
        const headers = Object.keys(state.fullData[0]);
        let csvContent = headers.join(',') + '\n';
        
        state.fullData.forEach(row => {
            const values = headers.map(header => {
                const val = row[header];
                // Escape commas or double quotes
                if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
                    return `"${val.replace(/"/g, '""')}"`;
                }
                return val;
            });
            csvContent += values.join(',') + '\n';
        });
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", `segmented_customers_k${state.k}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // TABLE VIEW RENDERING
    function renderTableHeaders() {
        if (state.fullData.length === 0) return;
        const columns = Object.keys(state.fullData[0]);
        
        let headerHtml = '';
        columns.forEach(col => {
            headerHtml += `<th>${col}</th>`;
        });
        elements.tableHeaders.innerHTML = headerHtml;
    }

    function applyTableFilters() {
        if (state.table.searchQuery.trim() === '') {
            state.filteredData = [...state.fullData];
            return;
        }
        
        const query = state.table.searchQuery.toLowerCase();
        state.filteredData = state.fullData.filter(row => {
            return Object.values(row).some(val => 
                String(val).toLowerCase().includes(query)
            );
        });
    }

    function renderTableBody() {
        const tbody = elements.tableBody;
        if (state.filteredData.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 30px;">No records match your query.</td></tr>`;
            elements.paginationInfo.textContent = "Showing 0 of 0 records";
            elements.btnPrevPage.disabled = true;
            elements.btnNextPage.disabled = true;
            return;
        }

        const startIndex = (state.table.currentPage - 1) * state.table.pageSize;
        const endIndex = Math.min(startIndex + state.table.pageSize, state.filteredData.length);
        const pageRecords = state.filteredData.slice(startIndex, endIndex);
        const columns = Object.keys(state.fullData[0]);
        
        let rowsHtml = '';
        pageRecords.forEach(row => {
            rowsHtml += '<tr>';
            columns.forEach(col => {
                const val = row[col];
                if (col === 'Cluster') {
                    rowsHtml += `<td><span class="cluster-tag cl-${val}">Cluster ${val}</span></td>`;
                } else {
                    rowsHtml += `<td>${val}</td>`;
                }
            });
            rowsHtml += '</tr>';
        });
        tbody.innerHTML = rowsHtml;

        // Update pagination details
        elements.paginationInfo.textContent = `Showing ${startIndex + 1} to ${endIndex} of ${state.filteredData.length} records`;
        
        elements.btnPrevPage.disabled = state.table.currentPage === 1;
        const maxPage = Math.ceil(state.filteredData.length / state.table.pageSize);
        elements.btnNextPage.disabled = state.table.currentPage >= maxPage;
    }

    // SEGMENTS VIEW RENDERING
    function renderSegmentBreakdown() {
        const container = elements.segmentsContainer;
        if (!state.clusteringResults || !state.clusteringResults.cluster_info) {
            container.innerHTML = `<div class="loading-spinner">No cluster model loaded.</div>`;
            return;
        }

        let segmentsHtml = '';
        state.clusteringResults.cluster_info.forEach(info => {
            const shortName = info.short_name;
            const template = archetypeTemplates[shortName] || archetypeTemplates["Medium Income, Medium Spend (Standard)"];
            
            const color = clusterColors[info.id % clusterColors.length];
            
            // Format labels income and spending values
            const x_label = state.dataset.featureNames[0].toLowerCase().includes('income') ? `$${info.avg_x.toFixed(1)}k` : info.avg_x.toFixed(1);
            const y_label = state.dataset.featureNames[1].toLowerCase().includes('score') ? `${info.avg_y.toFixed(1)}/100` : info.avg_y.toFixed(1);
            
            segmentsHtml += `
                <div class="segment-card" style="border-top: 4px solid ${color}">
                    <div class="segment-card-header">
                        <span class="segment-title">Cluster ${info.id}</span>
                        <span class="segment-pill ${template.pillClass}">${info.short_name}</span>
                    </div>
                    <div class="segment-card-body">
                        <div class="segment-stat-row">
                            <div class="segment-stat">
                                <strong class="segment-stat-val">${info.count}</strong>
                                <span class="segment-stat-lbl">Size</span>
                            </div>
                            <div class="segment-stat">
                                <strong class="segment-stat-val">${info.percentage.toFixed(1)}%</strong>
                                <span class="segment-stat-lbl">Ratio</span>
                            </div>
                            <div class="segment-stat">
                                <strong class="segment-stat-val">${info.avg_age.toFixed(0)} yrs</strong>
                                <span class="segment-stat-lbl">Avg Age</span>
                            </div>
                        </div>
                        
                        <div class="segment-stat-row" style="grid-template-columns: 1fr 1fr; margin-bottom: 20px;">
                            <div class="segment-stat">
                                <strong class="segment-stat-val" style="color: ${color}">${x_label}</strong>
                                <span class="segment-stat-lbl">Avg ${state.dataset.featureNames[0].split('(')[0]}</span>
                            </div>
                            <div class="segment-stat">
                                <strong class="segment-stat-val" style="color: ${color}">${y_label}</strong>
                                <span class="segment-stat-lbl">Avg ${state.dataset.featureNames[1].split('(')[0]}</span>
                            </div>
                        </div>

                        <p class="segment-desc">${template.desc}</p>
                        
                        <div class="segment-strategy ${template.class}">
                            <h6><i class="fa-solid fa-lightbulb"></i> Strategic Action Plan</h6>
                            <p>${template.campaign}</p>
                        </div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = segmentsHtml;
    }

    // CHARTS RENDERING (CHART.JS CONFIG)
    function renderElbowChart(labels, values) {
        if (state.charts.elbow) {
            state.charts.elbow.destroy();
        }

        const ctx = document.getElementById('elbowChart').getContext('2d');
        
        // Linear gradient for background glow
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(157, 78, 221, 0.2)');
        gradient.addColorStop(1, 'rgba(0, 242, 254, 0.02)');

        state.charts.elbow = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'WCSS (Inertia)',
                    data: values,
                    borderColor: '#00f2fe',
                    borderWidth: 3,
                    pointBackgroundColor: '#9d4edd',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    backgroundColor: gradient,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#ffffff',
                        bodyColor: '#00f2fe',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: function(context) {
                                return `WCSS: ${context.parsed.y.toLocaleString(undefined, {maximumFractionDigits: 0})}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: 'Number of Clusters (K)', color: '#6b7280', font: { size: 12, weight: 'bold' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: 'WCSS (Error)', color: '#6b7280', font: { size: 12, weight: 'bold' } }
                    }
                }
            }
        });
    }

    function renderScatterChart() {
        if (state.charts.scatter) {
            state.charts.scatter.destroy();
        }

        const ctx = document.getElementById('scatterChart').getContext('2d');
        const datasets = [];

        // 1. Compile datasets for clusters
        for (let i = 0; i < state.k; i++) {
            const clusterPoints = state.fullData
                .filter(row => row.Cluster === i)
                .map(row => ({
                    x: row[state.dataset.featureNames[0]],
                    y: row[state.dataset.featureNames[1]],
                    age: row['Age'] || 'N/A',
                    gender: row['Gender'] || 'N/A',
                    id: row['CustomerID'] || 'N/A'
                }));
                
            const info = state.clusteringResults.cluster_info.find(c => c.id === i);
            const label = info ? info.name : `Cluster ${i}`;

            datasets.push({
                label: label,
                data: clusterPoints,
                backgroundColor: clusterColors[i % clusterColors.length],
                pointRadius: 6,
                pointHoverRadius: 8,
                borderWidth: 1,
                borderColor: 'rgba(0,0,0,0.4)'
            });
        }

        // 2. Add Centroids dataset
        const centroidPoints = state.clusteringResults.centroids.map((coord, idx) => ({
            x: coord[0],
            y: coord[1],
            idx: idx
        }));

        datasets.push({
            label: 'Centroids',
            data: centroidPoints,
            backgroundColor: '#ffffff',
            borderColor: '#000000',
            borderWidth: 2.5,
            pointStyle: 'crossRot', // 'X' marker style
            pointRadius: 12,
            pointHoverRadius: 14,
            showLine: false
        });

        // 3. Render Chart
        state.charts.scatter = new Chart(ctx, {
            type: 'scatter',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Inter', size: 11 },
                            padding: 12,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: '#111827',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                const pt = context.raw;
                                if (context.dataset.label === 'Centroids') {
                                    return `Centroid ${pt.idx}: (${state.dataset.featureNames[0].split('(')[0]}: ${pt.x.toFixed(1)}, ${state.dataset.featureNames[1].split('(')[0]}: ${pt.y.toFixed(1)})`;
                                }
                                return [
                                    `Customer ID: ${pt.id}`,
                                    `Gender: ${pt.gender}, Age: ${pt.age}`,
                                    `${state.dataset.featureNames[0].split('(')[0]}: ${pt.x}`,
                                    `${state.dataset.featureNames[1].split('(')[0]}: ${pt.y}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: state.dataset.featureNames[0], color: '#6b7280', font: { size: 12, weight: 'bold' } }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter' } },
                        title: { display: true, text: state.dataset.featureNames[1], color: '#6b7280', font: { size: 12, weight: 'bold' } }
                    }
                }
            }
        });
    }

    // Boot App
    init();
});
