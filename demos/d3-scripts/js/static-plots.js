// Generate static publication-ready plots
function generateStaticPlot(plotType) {
    switch(plotType) {
        case 'spectrum':
            createStaticSpectrum();
            break;
        case 'properties':
            createStaticProperties();
            break;
        case 'radar':
            createStaticRadar();
            break;
        case 'network':
            createStaticNetwork();
            break;
        case 'heatmap':
            createStaticHeatmap();
            break;
        case 'timeseries':
            createStaticTimeSeries();
            break;
    }
}

function createStaticSpectrum() {
    // Create a static version optimized for publication
    const margin = {top: 40, right: 100, bottom: 60, left: 80};
    const width = 600 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Remove existing static plot
    d3.select("#static-spectrum").remove();
    
    const svg = d3.select("body")
        .append("svg")
        .attr("id", "static-spectrum")
        .attr("class", "publication-plot")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .style("background", "white");

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Load data and create plot
    d3.json("hardware_data_1758140958.json").then(data => {
        const measurements = data.led_spectroscopy.measurements;
        
        // Color scheme for publication
        const colors = ["#1f77b4", "#ff7f0e", "#2ca02c"];
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain(d3.extent(measurements[0].spectrum_wavelengths))
            .range([0, width]);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(measurements, d => d3.max(d.spectrum_intensities))])
            .range([height, 0]);

        // Line generator
        const line = d3.line()
            .x(d => xScale(d.wavelength))
            .y(d => yScale(d.intensity))
            .curve(d3.curveMonotoneX);

        // Add grid
        g.append("g")
            .attr("class", "grid")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale)
                .tickSize(-height)
                .tickFormat("")
            )
            .selectAll("line")
            .attr("stroke", "#e0e0e0")
            .attr("stroke-width", 0.5);

        g.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(yScale)
                .tickSize(-width)
                .tickFormat("")
            )
            .selectAll("line")
            .attr("stroke", "#e0e0e0")
            .attr("stroke-width", 0.5);

        // Plot lines
        measurements.forEach((measurement, i) => {
            const lineData = measurement.spectrum_wavelengths.map((wavelength, j) => ({
                wavelength: wavelength,
                intensity: measurement.spectrum_intensities[j]
            }));

            g.append("path")
                .datum(lineData)
                .attr("fill", "none")
                .attr("stroke", colors[i])
                .attr("stroke-width", 2)
                .attr("d", line);
        });

        // Axes
        g.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .style("font-family", "Times New Roman")
            .style("font-size", "12px");

        g.append("g")
            .call(d3.axisLeft(yScale))
            .selectAll("text")
            .style("font-family", "Times New Roman")
            .style("font-size", "12px");

        // Labels
        g.append("text")
            .attr("class", "axis-label")
            .attr("x", width / 2)
            .attr("y", height + 50)
            .style("text-anchor", "middle")
            .style("font-family", "Times New Roman")
            .style("font-size", "14px")
            .style("font-weight", "bold")
            .text("Wavelength (nm)");

        g.append("text")
            .attr("class", "axis-label")
            .attr("transform", "rotate(-90)")
            .attr("x", -height / 2)
            .attr("y", -60)
            .style("text-anchor", "middle")
            .style("font-family", "Times New Roman")
            .style("font-size", "14px")
            .style("font-weight", "bold")
            .text("Intensity (a.u.)");

        // Title
        g.append("text")
            .attr("class", "title")
            .attr("x", width / 2)
            .attr("y", -20)
            .style("text-anchor", "middle")
            .style("font-family", "Times New Roman")
            .style("font-size", "16px")
            .style("font-weight", "bold")
            .text("LED Spectroscopy Analysis");

        // Legend
        const legend = g.append("g")
            .attr("transform", `translate(${width + 20}, 20)`);

        measurements.forEach((measurement, i) => {
            const legendRow = legend.append("g")
                .attr("transform", `translate(0, ${i * 20})`);

            legendRow.append("line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", 0)
                .attr("y2", 0)
                .attr("stroke", colors[i])
                .attr("stroke-width", 2);

            legendRow.append("text")
                .attr("x", 25)
                .attr("y", 4)
                .style("font-family", "Times New Roman")
                .style("font-size", "12px")
                .text(`${measurement.center_wavelength_nm} nm`);
        });

        // Auto-export
        setTimeout(() => {
            exportSVG("static-spectrum", "led_spectroscopy_publication", true);
        }, 100);
    });
}

function generateStatic3D(containerId) {
    const container = document.getElementById(containerId);
    const renderer = container._threeRenderer;
    const scene = container._threeScene;
    const camera = container._threeCamera;
    
    if (!renderer) {
        console.error('No 3D renderer found');
        return;
    }
    
    // Render at high resolution for publication
    const originalSize = renderer.getSize(new THREE.Vector2());
    renderer.setSize(1200, 900); // High resolution
    
    // Render the scene
    renderer.render(scene, camera);
    
    // Get image data
    const canvas = renderer.domElement;
    const dataURL = canvas.toDataURL('image/png');
    
    // Create download link
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = `${containerId}_publication_3d.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Restore original size
    renderer.setSize(originalSize.x, originalSize.y);
}
