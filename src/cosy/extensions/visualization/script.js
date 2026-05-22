// ============ Configuration & Layout ============
const margin = { top: 20, right: 120, bottom: 20, left: 120 };
const canvas_height = 700;
const canvas_width = 960;
const tree_width = canvas_width - margin.right - margin.left;
const tree_height = canvas_height - margin.top - margin.bottom;
const tree_level_depth = 180;
const node_size = 25;
const preview_node_size = 5;
const transition_duration = 750;

// ============ Global State ============
let nodeIdCounter = 0;
let jsonData;
let root;
let selectedResult = 0;
let stripeIdCounter = 0;
let allStripeGradients = [];
let colorMap = {};
let gradientCache = {}; // Maps color array JSON to gradient ID



// Create sidebar container
const sidebar = d3.select("body").append("div")
    .style("position", "fixed")
    .style("left", "0")
    .style("top", "0")
    .style("width", "200px")
    .style("height", "100vh")
    .style("background-color", "#f5f5f5")
    .style("padding", "20px")
    .style("box-sizing", "border-box")
    .style("overflow-y", "auto")
    .style("border-right", "1px solid #ddd");

// Adjust body margin to account for sidebar
d3.select("body")
    .style("margin", "0")
    .style("padding", "0");

// Result selector
const resultSelector = sidebar.append("div")
    .style("margin-bottom", "20px");

resultSelector.append("div")
    .style("font-weight", "bold")
    .style("margin-bottom", "10px")
    .text("Result:");

resultSelector.append("button")
    .text("−")
    .style("margin-right", "5px")
    .on("click", () => {
        if (selectedResult > 0) {
            refreshSelectedResult(selectedResult - 1);
        }
    });

const selectedResultInput = resultSelector.append("input")
    .attr("type", "number")
    .attr("class", "counter")
    .attr("min", "0")
    .style("margin", "0 5px")
    .style("width", "40px")
    .style("padding", "4px")
    .property("value", selectedResult)
    .on("change", function() {
        const newValue = parseInt(this.value);
        if (!isNaN(newValue)) {
            refreshSelectedResult(newValue);
        }
    });

// Add CSS to hide number input spinners
d3.select("head").append("style")
    .text(`
        input.counter::-webkit-outer-spin-button,
        input.counter::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        input.counter {
            -moz-appearance: textfield;
        }
    `);

resultSelector.append("button")
    .text("+")
    .style("margin-left", "5px")
    .on("click", () => {
        refreshSelectedResult(selectedResult + 1);
    });

// Checkboxes
// ============ Checkbox State ============
const checkboxState = {
    showCombinatorNames: true,
    showValues: true,
    showParameters: true,
    hideParameters: false
};

const checkboxContainer = sidebar.append("div");

checkboxContainer.append("div")
    .style("font-weight", "bold")
    .style("margin-bottom", "10px")
    .text("Display Options:");

const checkboxes = [
    { id: "showCombinatorNames", label: "Show Combinator Names", variable: "showCombinatorNames", initiallyChecked: true },
    { id: "showValues", label: "Show Values", variable: "showValues", initiallyChecked: true },
    { id: "showParameters", label: "Show Parameters", variable: "showParameters", initiallyChecked: true },
    { id: "hideParameters", label: "Hide Parameter Nodes", variable: "hideParameters", initiallyChecked: false, reloadAll: true }
];

checkboxes.forEach((checkbox) => {
    const checkboxGroup = checkboxContainer.append("label")
        .style("display", "block")
        .style("margin-bottom", "8px")
        .style("cursor", "pointer");
    
    const input = checkboxGroup.append("input")
        .attr("type", "checkbox")
        .attr("id", checkbox.id)
        .property("checked", checkbox.initiallyChecked)
        .style("margin-right", "8px");
    
    input.on("change", function() {
        checkboxState[checkbox.variable] = this.checked;
        console.log(checkbox.variable + " = " + checkboxState[checkbox.variable]);
        if (checkbox.reloadAll) {
            loadResult(selectedResult);
        } else {
            update(root);
        }
    });
    
    checkboxGroup.append("text")
        .text(checkbox.label);
});

// Color map display
const colorMapContainer = sidebar.append("div")
    .style("margin-top", "20px");

colorMapContainer.append("div")
    .style("font-weight", "bold")
    .style("margin-bottom", "10px")
    .text("Constructor colors:");

const colorMapDisplay = colorMapContainer.append("div")
    .attr("id", "colorMapDisplay")
    .style("font-size", "12px");

function refreshColorMapDisplay() {
    colorMapDisplay.selectAll("*").remove();
    Object.entries(colorMap).forEach(([key, value]) => {
        // value is now an array of color strings
        const gradId = getOrCreateGradient(value);
        
        const entry = colorMapDisplay.append("div")
            .style("margin-bottom", "8px")
            .style("position", "relative")
            .style("width", "100%")
            .style("height", "25px");
        
        // Create an inline SVG element with the gradient pattern
        const svgElement = entry.append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .style("border", "1px solid #ccc")
            .style("position", "absolute")
            .style("top", "0")
            .style("left", "0");
        
        // Add defs and pattern for this specific box
        const defs = svgElement.append("defs");
        const pattern = defs.append("pattern")
            .attr("id", gradId + "_box")
            .attr("x", "0")
            .attr("y", "0")
            .attr("width", "1")
            .attr("height", "1")
            .attr("patternUnits", "objectBoundingBox")
            .attr("viewBox", "0 0 " + value.length + " 1")
            .attr("preserveAspectRatio", "none");
        
        value.forEach((colorArray, stripeIndex) => {
            const colorGradId = getOrCreateGradient([colorArray]);
            const stripeWidth = 1;
            const stripeX = stripeIndex;
            
            pattern.append("rect")
                .attr("x", stripeX)
                .attr("y", "0")
                .attr("width", stripeWidth)
                .attr("height", "1")
                .style("fill", "url(#" + colorGradId + ")");
        });
        
        // Fill the SVG with the pattern
        svgElement.append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .style("fill", "url(#" + gradId + "_box)");
        
        // Add text label on top of the SVG
        entry.append("text")
            .style("position", "absolute")
            .style("top", "50%")
            .style("left", "50%")
            .style("transform", "translate(-50%, -50%)")
            .style("font-size", "10px")
            .style("font-weight", "bold")
            .style("color", "#000")
            .style("pointer-events", "none")
            .style("z-index", "1")
            .text(key);
    });
}

function refreshSelectedResult(newResult) {
    oldResult = selectedResult;
    selectedResult = newResult;
    const successful = loadResult(selectedResult);
    console.log("Selected result: " + selectedResult + " (successful: " + successful + ")");
    if (successful) {
        selectedResultInput.property("value", selectedResult);
        refreshColorMapDisplay();
    } else {
        selectedResult = oldResult;
        selectedResultInput.property("value", selectedResult);
    }
}

// ============ D3 Setup ============
const tree = d3.layout.tree()
    .size([tree_height, tree_width]);

const diagonal = d3.svg.diagonal()
    .projection((d) => {
        //console.log("Diagonal projection:", d);
        return [d.y, d.x];
    });

const realSvg = d3.select("body").append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .style("margin-left", "200px")
    .style("background-color", "#f2f2f2");

const svgDefs = realSvg.append("defs");
const svg = realSvg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// ============ Pan and Zoom Setup ============
const zoom = d3.behavior.zoom()
    .on("zoom", () => {
        svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    });

realSvg.call(zoom);

// Load visualization data
d3.json("./results.json", (error, data) => {
    jsonData = data;
    loadResult(selectedResult);
    refreshColorMapDisplay();
});

function filterParameterNodes(node) {
    if (!node) return node;
    
    // Filter out parameter nodes if hideParameters is checked
    if (checkboxState.hideParameters && !node.is_combinator) {
        return null;
    }
    
    // Recursively filter children
    if (node.children) {
        node.children = node.children
            .map(filterParameterNodes)
            .filter(child => child !== null);
    }
    
    if (node._children) {
        node._children = node._children
            .map(filterParameterNodes)
            .filter(child => child !== null);
    }
    
    return node;
}

function togglePreview(node) {
    if (node.only_preview) {
        node.only_preview = false;
    }
    else {
        node.only_preview = true;
    }
}

function loadResult(resultNum) {
    let resultData;
    try {
        resultData = JSON.parse(JSON.stringify(jsonData[resultNum]));
    } catch (e) {
        return false;
    }
    if (resultData === undefined) {
        return false;
    }
    let treeData = resultData.tree;
    colorMap = resultData.color_map;
    root = filterParameterNodes(treeData);
    if (root === null) {
        return false;
    }
    
    root.x0 = tree_height / 2;
    root.y0 = 0;
    root.y = 0;
    //console.log("Initial root:", root);
    //console.log("root_y:", root.y);

    // Recursively collapse all children initially
    function collapse(d) {
        d.collapsed = true;
        d.only_preview = true;
        if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null;
        }
    }
    
    if (root.children) {
        root.children.forEach(collapse);
    }
    root.collapsed = true;
    root.only_preview = false;
    
    update(root);
    return true;
}

// ============ Stripe Gradient Management ============
function clearStripes() {
    allStripeGradients = [];
}

function getOrCreateGradient(colorArray) {
    // Create a deterministic ID based on the color array
    const colorKey = JSON.stringify(colorArray);
    
    // Check if gradient already exists in cache
    if (gradientCache[colorKey]) {
        return gradientCache[colorKey];
    }
    
    // Create a new gradient
    const gradId = "colorGradient_" + Object.keys(gradientCache).length;
    const grad = svgDefs.append("linearGradient")
        .attr("id", gradId)
        .attr("spreadMethod", "pad")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "0%")
        .attr("y2", "100%")
        .attr("gradientUnits", "objectBoundingBox")
        .attr("patternTransform", "rotate(45)");
    
    const colorStep = 1 / colorArray.length;
    let colorOffset = 0;
    
    colorArray.forEach((color) => {
        grad.append("stop")
            .attr("offset", colorOffset)
            .attr("stop-color", color);
        colorOffset += colorStep;
        grad.append("stop")
            .attr("offset", colorOffset)
            .attr("stop-color", color);
    });
    
    // Cache it
    gradientCache[colorKey] = gradId;
    allStripeGradients.push(grad);
    
    return gradId;
}

function makeStripes(id, colors) {
    console.log("Making stripes with id:", id, "and colors:", colors);
    const numStripes = colors.length;
    
    // Create a pattern using a viewBox to maintain coordinate system
    const pattern = svgDefs.append("pattern")
        .attr("id", id)
        .attr("x", "0")
        .attr("y", "0")
        .attr("width", "1")
        .attr("height", "1")
        .attr("patternUnits", "objectBoundingBox")
        .attr("viewBox", "0 0 " + numStripes + " 1")
        .attr("preserveAspectRatio", "none")
        .attr("patternTransform", "rotate(45)");
    
    colors.forEach((colorArray, stripeIndex) => {
        const stripeWidth = 1;
        const stripeX = stripeIndex;
        
        // Get or create gradient for this color array
        const gradId = getOrCreateGradient(colorArray);
        
        // Add rectangle for this stripe
        pattern.append("rect")
            .attr("x", stripeX)
            .attr("y", "0")
            .attr("width", stripeWidth)
            .attr("height", "1")
            .style("fill", "url(#" + gradId + ")");
    });
}
// Example call:[["green", "blue", "yellow"], ["red"]]
// previously: makeStripes("myStripes", [["green", "blue"], ["red"]]);
//makeStripes("myStripes", [["green", "blue", "yellow"], ["red"]]);

// ============ Tree Rendering ============
function update(source) {
    // Compute the new tree layout
    let nodes = tree.nodes(root);
    let links = tree.links(nodes);

    // Filter out parameter nodes if hideParameters is checked
    if (checkboxState.hideParameters) {
        nodes = nodes.filter(d => d.is_combinator);
        links = links.filter(link => link.source.is_combinator && link.target.is_combinator);
    }
    source.y = 0;
    // Normalize for fixed-depth
    nodes.forEach((d) => {d.y = (d.parent.y || 0) + 100 + d.val.length*10});
    // { d.y = d.depth * tree_level_depth; console.log(d.parent)});

    // Set unique ID for each node
    const node = svg.selectAll("g.node")
        .data(nodes, (d) => d.id || (d.id = ++nodeIdCounter));

    // Enter any new nodes at the parent's previous position
    const newNodes = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", (d) => "translate(" + (d.parent ? d.parent.y0 : source.y0) + "," + (d.parent ? d.parent.x0 : source.x0) + ")")
        .on("click", click)
        .on("mouseover", function() {
            // Emphasize labels on hovered node
            d3.select(this).selectAll("text")
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .style("display", "inline");
            
            // Hide labels on all other nodes
            svg.selectAll("g.node:not(:hover)").selectAll("text")
                .style("display", "none");
        })
        .on("mouseout", function() {
            // Restore labels based on checkbox state
            svg.selectAll("g.node").selectAll("text")
                .style("font-size", "12px")
                .style("font-weight", "normal");
            
            svg.select("g.node:hover").selectAll("text")
                .style("display", (d, i) => {
                    if (d.only_preview) return "none";
                    if (i === 0) return checkboxState.showValues ? "inline" : "none";
                    if (i === 1) return checkboxState.showParameters ? "inline" : "none";
                    if (i === 2) return checkboxState.showCombinatorNames ? "inline" : "none";
                    return "none";
                });
            
            // Restore visibility for non-hovered nodes based on checkbox state
            svg.selectAll("g.node").each(function(d) {
                if (this !== d3.event.target && d3.event.target !== d3.event.relatedTarget) {
                    d3.select(this).select("text.value-text")
                        .style("display", (d) => checkboxState.showValues && !d.only_preview ? "inline" : "none");
                    d3.select(this).select("text.parameter-text")
                        .style("display", (d) => checkboxState.showParameters && !d.only_preview ? "inline" : "none");
                    d3.select(this).select("text.combinator-text")
                        .style("display", (d) => checkboxState.showCombinatorNames && !d.only_preview ? "inline" : "none");
                }
            });
        });
    

    node.each(function(d) {
        d.x_pos = d.x;
        if (d.only_preview && d.parent) {
            d.x_pos -= (d.x - d.parent.x) * 0.8;
        }
    });
    // Add circle for combinators or rect for parameters
    newNodes.each(function(d) {
        //console.log("Node5:", d);
        const nodeGroup = d3.select(this);
        if (d.is_combinator) {
            nodeGroup.append("circle")
                .attr("r", 1e-5)
                .style("fill", () => {
                    stripeIdCounter += 1;
                    const stripeName = "stripe" + d.id + stripeIdCounter;
                    //makeStripes(stripeName, [["green", "blue", "yellow"], ["red", "black"]]);
                    makeStripes(stripeName, d.colors);
                    return "url(#" + stripeName + ")";
                });
        } else {
            nodeGroup.append("rect")
                .attr("x", 1e-5)
                .attr("y", 1e-5)
                .attr("width", 1e-5)
                .attr("height", 1e-5)
                .style("fill", "#a2a2a2")
                .style("stroke", "#878787")
                .style("stroke-width", 2);
        }
    });

    // Add value label (left of node)
    newNodes.append("text")
        .attr("class", "value-text")
        .attr("x", -1.1 * node_size)
        .attr("dy", ".35em")
        .attr("text-anchor", "end")
        .attr()
        .text((d) => d.val);
    
    // Add parameter label (above node)
    newNodes.append("text")
        .attr("class", "parameter-text")
        .attr("x", 0)
        .attr("dy", -1.1 * node_size)
        .attr("text-anchor", "middle")
        .text((d) => d.parameter)
        .style("text-anchor", "middle");
    
    // Add combinator label (below node)
    newNodes.append("text")
        .attr("class", "combinator-text")
        .attr("x", 0)
        .attr("dy", 1.3 * node_size)
        .attr("text-anchor", "middle")
        .text((d) => d.combinator)

    // node.each(function(d) {
    //     if (d.only_preview) {
    //         d.selectAll("text").style("display", "none");
    //     } else {
    //         d3.select(this).select("text.value-text")
    //             .style("display", checkboxState.showValues ? "inline" : "none");
    //         d3.select(this).select("text.parameter-text")
    //             .style("display", checkboxState.showParameters ? "inline" : "none");
    //         d3.select(this).select("text.combinator-text")
    //             .style("display", checkboxState.showCombinatorNames ? "inline" : "none");
    //     }
    // });
    // Reset stripe counter and add default stripes
    // makeStripes("myStripes", ["green", "red"]);
    stripeIdCounter = 0;
    clearStripes();
    
    
    // Transition nodes to their new position
    const movedNode = node.transition().duration(transition_duration)
        .attr("transform", (d) => "translate(" + d.y + "," + d.x_pos + ")");
    
    movedNode.select("circle")
        .attr("r", (d) => d.only_preview ? preview_node_size : node_size);
    
    movedNode.select("rect")
        .attr("x", (d) =>  -0.75 * (d.only_preview ? preview_node_size : node_size))
        .attr("y", (d) => -0.75 * (d.only_preview ? preview_node_size : node_size))
        .attr("width", (d) => 1.5 * (d.only_preview ? preview_node_size : node_size))
        .attr("height", (d) => 1.5 * (d.only_preview ? preview_node_size : node_size));
    
    movedNode.select("text.value-text")
        .style("display", (d) => checkboxState.showValues && !d.only_preview ? "inline" : "None");
    
    movedNode.select("text.parameter-text")
        .style("display", (d) => checkboxState.showParameters && !d.only_preview ? "inline" : "None");
    
    movedNode.select("text.combinator-text")
        .style("display", (d) => checkboxState.showCombinatorNames && !d.only_preview ? "inline" : "None");

    // Transition exiting nodes to the parent's new position
    const hiddenNodes = node.exit()
        ;//.remove();

    const fullyRemovedNodes = hiddenNodes.remove();
    //const previewNodes = hiddenNodes.filter((d) => d.parent === source);
    fullyRemovedNodes.select("circle")
        .attr("r", 1e-5);
    
    fullyRemovedNodes.select("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 0)
        .attr("height", 0);
    //console.log("previewNodes", previewNodes);
    //previewNodes.selectAll("text").forEach((t) => console.log("text", t));
    //previewNodes.transition().duration(transition_duration)
    //    .attr("transform", (d) => "translate(" + (source.y + 50) + "," + d.x + ")")
    //previewNodes.selectAll("text")
    //    .style("display", "None").forEach((t) => console.log("text", t));
    //previewNodes.select("text.value-text")
    //    .attr("display", "none");

    // Update the links
    const link = svg.selectAll("path.link")
        .data(links, (d) => d.target.id);

    // Enter any new links at the parent's previous position
    link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", (d) => {
            const o = { x: d.source.x0, y: d.source.y0 };
            return diagonal({ source: o, target: o });
        })
        .append("svg:title")
            .text((d) => d.target.edge_name);

    // Transition links to their new position
    link.transition().duration(transition_duration)
        .attr("d", (d) => {
            //console.log("Link transition:", d);
            const from = { x: d.source.x_pos, y: d.source.y };
            const to = { x: d.target.x_pos, y: d.target.y };
            const o = { x: d.x, y: d.y };
            return diagonal({ source: from, target: to }); //{ x: d.x_pos, y: d.y });
        });

    // Transition exiting links to the parent's new position
    link.exit().transition().duration(transition_duration)
        .attr("d", (d) => {
            const o = { x: source.x_pos, y: source.y };
            return diagonal({ source: o, target: o });
        })
        .remove();

    // Stash the old positions for transition
    nodes.forEach((d) => {
        d.x0 = d.x;
        d.y0 = d.y;
    });
}
// Toggle children on click
function click(d) {
    if (d.only_preview)
        return;
    if (d.collapsed) {
        d.collapsed = false;
        for (c of d.children) {
            c.only_preview = false;
            c.children = c._children;
            c._children = null;
        }
    } else {
        d.collapsed = true;
        for (c of d.children) {
            c.only_preview = true;
            c._children = c.children;
            c.children = null;
        }
    }
    update(d);
}