// Async data loader with progress tracking and error handling
class DataLoader {
    constructor() {
        this.loadingProgress = {};
        this.totalFiles = 4;
        this.loadedFiles = 0;
    }

    // Show loading progress
    updateProgress(fileName, progress) {
        this.loadingProgress[fileName] = progress;
        const totalProgress = Object.values(this.loadingProgress).reduce((a, b) => a + b, 0) / this.totalFiles;
        this.showProgress(totalProgress, fileName);
    }

    showProgress(progress, currentFile) {
        const progressBar = document.getElementById('loading-progress');
        const progressText = document.getElementById('loading-text');
        
        if (progressBar) {
            progressBar.style.width = `${progress * 100}%`;
        }
        if (progressText) {
            progressText.textContent = `Loading ${currentFile}... ${Math.round(progress * 100)}%`;
        }
    }

    // Fetch with progress tracking
    async fetchWithProgress(url, fileName) {
        try {
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const contentLength = response.headers.get('content-length');
            if (!contentLength) {
                // If no content-length, just return the json
                this.updateProgress(fileName, 1);
                return await response.json();
            }

            const total = parseInt(contentLength, 10);
            let loaded = 0;

            const reader = response.body.getReader();
            const chunks = [];

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                chunks.push(value);
                loaded += value.length;
                
                this.updateProgress(fileName, loaded / total);
            }

            // Combine chunks and parse JSON
            const allChunks = new Uint8Array(loaded);
            let position = 0;
            for (const chunk of chunks) {
                allChunks.set(chunk, position);
                position += chunk.length;
            }

            const text = new TextDecoder().decode(allChunks);
            return JSON.parse(text);

        } catch (error) {
            console.error(`Error loading ${fileName}:`, error);
            throw error;
        }
    }

    // Load all data files asynchronously
    async loadAllData() {
        const dataFiles = [
            { url: './data/hardware_data_1758140958.json', key: 'hardware' },
            { url: './data/molecular_data_1758140958.json', key: 'molecular' },
            { url: './data/network_data_1758140958.json', key: 'network' },
            { url: './data/timeseries_data_1758140958.json', key: 'timeseries' }
        ];

        try {
            // Show loading UI
            this.showLoadingUI();

            // Load all files concurrently with progress tracking
            const dataPromises = dataFiles.map(async (file) => {
                const data = await this.fetchWithProgress(file.url, file.key);
                return { key: file.key, data };
            });

            const results = await Promise.all(dataPromises);

            // Convert array to object
            const globalData = {};
            results.forEach(result => {
                globalData[result.key] = result.data;
            });

            this.hideLoadingUI();
            return globalData;

        } catch (error) {
            this.hideLoadingUI();
            this.showError(error);
            throw error;
        }
    }

    // Alternative: Load files sequentially (for slower connections)
    async loadDataSequentially() {
        const dataFiles = [
            { url: './data/hardware_data_1758140958.json', key: 'hardware' },
            { url: './data/molecular_data_1758140958.json', key: 'molecular' },
            { url: './data/network_data_1758140958.json', key: 'network' },
            { url: './data/timeseries_data_1758140958.json', key: 'timeseries' }
        ];

        const globalData = {};
        this.showLoadingUI();

        try {
            for (const file of dataFiles) {
                console.log(`Loading ${file.key}...`);
                globalData[file.key] = await this.fetchWithProgress(file.url, file.key);
                console.log(`✓ ${file.key} loaded successfully`);
            }

            this.hideLoadingUI();
            return globalData;

        } catch (error) {
            this.hideLoadingUI();
            this.showError(error);
            throw error;
        }
    }

    showLoadingUI() {
        const loadingHTML = `
            <div id="loading-overlay" style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.8);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                color: white;
                font-family: Arial, sans-serif;
            ">
                <div style="text-align: center;">
                    <div style="margin-bottom: 20px;">
                        <div style="font-size: 24px; margin-bottom: 10px;">Loading Data...</div>
                        <div id="loading-text" style="font-size: 14px;">Initializing...</div>
                    </div>
                    <div style="width: 300px; height: 20px; background: #333; border-radius: 10px; overflow: hidden;">
                        <div id="loading-progress" style="
                            width: 0%;
                            height: 100%;
                            background: linear-gradient(90deg, #007bff, #0056b3);
                            transition: width 0.3s ease;
                        "></div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', loadingHTML);
    }

    hideLoadingUI() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    showError(error) {
        const errorHTML = `
            <div id="error-overlay" style="
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.9);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                color: white;
                font-family: Arial, sans-serif;
            ">
                <div style="text-align: center; max-width: 500px; padding: 20px;">
                    <h2 style="color: #ff4444; margin-bottom: 20px;">Error Loading Data</h2>
                    <p style="margin-bottom: 20px;">${error.message}</p>
                    <button onclick="location.reload()" style="
                        background: #007bff;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 5px;
                        cursor: pointer;
                        font-size: 16px;
                    ">Retry</button>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', errorHTML);
    }
}
