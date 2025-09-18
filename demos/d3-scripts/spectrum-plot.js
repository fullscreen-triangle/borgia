function createSpectrumPlot(data) {
    const margin = {top: 20, right: 80, bottom: 50, left: 60};
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const svg = d3.select("#spectrum-plot")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Color scale for different center wavelengths
    const colorScale = d3.scaleOrdinal()
        .domain([470, 525, 625])
        .range(["#4169E1", "#32CD32", "#DC143C"]);

    // Scales
    const xScale = d3.scaleLinear()
        .domain(d3.extent(data.led_spectroscopy.measurements[0].spectrum_wavelengths))
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([0, d3.max(data.led_spectroscopy.measurements, d => 
            d3.max(d.spectrum_intensities))])
        .range([height, 0]);

    // Line generator
    const line = d3.line()
        .x(d => xScale(d.wavelength))
        .y(d => yScale(d.intensity))
        .curve(d3.curveMonotoneX);

    // Add axes
    g.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append("text")
        .attr("x", width / 2)
        .attr("y", 40)
        .attr("fill", "black")
        .style("text-anchor", "middle")
        .text("Wavelength (nm)");

    g.append("g")
        .call(d3.axisLeft(yScale))
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -40)
        .attr("x", -height / 2)
        .attr("fill", "black")
        .style("text-anchor", "middle")
        .text("Intensity");

    // Plot lines for each measurement
    data.led_spectroscopy.measurements.forEach(measurement => {
        const lineData = measurement.spectrum_wavelengths.map((wavelength, i) => ({
            wavelength: wavelength,
            intensity: measurement.spectrum_intensities[i]
        }));

        g.append("path")
            .datum(lineData)
            .attr("fill", "none")
            .attr("stroke", colorScale(measurement.center_wavelength_nm))
            .attr("stroke-width", 2)
            .attr("d", line);

        // Add legend
        g.append("text")
            .attr("x", width + 10)
            .attr("y", measurement.center_wavelength_nm === 470 ? 20 : 
                     measurement.center_wavelength_nm === 525 ? 40 : 60)
            .attr("fill", colorScale(measurement.center_wavelength_nm))
            .text(`${measurement.center_wavelength_nm}nm`);
    });

    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([1, 10])
        .on("zoom", function(event) {
            const newXScale = event.transform.rescaleX(xScale);
            const newYScale = event.transform.rescaleY(yScale);
            
            g.select(".x-axis").call(d3.axisBottom(newXScale));
            g.select(".y-axis").call(d3.axisLeft(newYScale));
            
            g.selectAll("path").attr("d", line
                .x(d => newXScale(d.wavelength))
                .y(d => newYScale(d.intensity)));
        });

    svg.call(zoom);
}
