// Universal SVG export with publication-ready styling
function exportSVG(containerId, filename, isPublication = false) {
    const container = document.getElementById(containerId);
    const svgElement = container.querySelector('svg');
    
    if (!svgElement) {
        console.error('No SVG found in container:', containerId);
        return;
    }

    // Clone SVG to avoid modifying original
    const clonedSvg = svgElement.cloneNode(true);
    
    // Add publication-ready styles
    const style = document.createElement('style');
    style.textContent = `
        ${isPublication ? `
        text { 
            font-family: 'Times New Roman', serif !important; 
            font-size: 12px !important;
        }
        .axis-label { 
            font-size: 14px !important; 
            font-weight: bold !important; 
        }
        .title { 
            font-size: 16px !important; 
            font-weight: bold !important; 
        }
        ` : `
        text { 
            font-family: Arial, sans-serif; 
        }
        `}
        .axis path, .axis line { 
            fill: none; 
            stroke: #000; 
            shape-rendering: crispEdges; 
        }
        .grid-line { 
            stroke: #e0e0e0; 
            stroke-dasharray: 3,3; 
        }
        .tooltip {
            pointer-events: none;
        }
    `;
    clonedSvg.insertBefore(style, clonedSvg.firstChild);
    
    // Set proper dimensions and viewBox
    const bbox = svgElement.getBBox();
    clonedSvg.setAttribute('viewBox', `0 0 ${bbox.width + bbox.x} ${bbox.height + bbox.y}`);
    clonedSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    
    // Serialize and download
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(clonedSvg);
    
    const blob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}_${isPublication ? 'publication' : 'interactive'}.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// 3D SVG export using SVG renderer
function export3DSVG(containerId, filename) {
    const container = document.getElementById(containerId);
    const canvas = container.querySelector('canvas');
    
    if (!canvas) {
        console.error('No 3D canvas found in container:', containerId);
        return;
    }
    
    // Convert canvas to SVG (this is a simplified approach)
    // For true 3D SVG export, we'd need to implement SVG renderer
    const dataURL = canvas.toDataURL('image/png');
    
    // Create SVG with embedded image
    const svg = `
        <svg xmlns="http://www.w3.org/2000/svg" 
             xmlns:xlink="http://www.w3.org/1999/xlink"
             width="${canvas.width}" 
             height="${canvas.height}">
            <image xlink:href="${dataURL}" 
                   width="${canvas.width}" 
                   height="${canvas.height}"/>
        </svg>
    `;
    
    const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}_3d.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Data processing utilities
function normalizeProperty(data, property, accessor = d => d) {
    const values = data.map(accessor);
    const extent = d3.extent(values);
    return values.map(v => (v - extent[0]) / (extent[1] - extent[0]));
}

function calculateNetworkMetrics(adjacencyMatrix) {
    const n = adjacencyMatrix.length;
    const metrics = {
        degrees: [],
        clustering: [],
        centrality: []
    };
    
    // Calculate degree for each node
    for (let i = 0; i < n; i++) {
        metrics.degrees.push(d3.sum(adjacencyMatrix[i]));
    }
    
    return metrics;
}
