// ============ Configuration & Layout ============
const margin = { top: 20, right: 120, bottom: 20, left: 120 };
const canvas_height = 700;
const canvas_width = 960;
const tree_width = canvas_width - margin.right - margin.left;
const tree_height = canvas_height - margin.top - margin.bottom;
const tree_level_depth = 180;
const circle_radius = 25;
const transition_duration = 750;

// ============ Global State ============
let nodeIdCounter = 0;
let treeData;
let root;
let selectedResult = 0;
let stripeIdCounter = 0;
let allStripeGradients = [];



const resultSelector = d3.select("body").append("div");

resultSelector.append("button")
    .text("−")
    .on("click", () => {
        if (selectedResult > 1) {
            refreshSelectedResult(-1);
        }
    });

const selectedResultText = resultSelector.append("span")
    .attr("class", "counter")
    .text(selectedResult);

resultSelector.append("button")
    .text("+")
    .on("click", () => {
        refreshSelectedResult(1);
    });

// Add checkboxes
// ============ Checkbox State ============
const checkboxState = {
    showCombinatorNames: true,
    showValues: true,
    showParameters: true,
    hideParameters: false
};

const checkboxContainer = d3.select("body")
    .append("div")
    .style("margin-top", "10px")
    .style("margin-bottom", "10px");

const checkboxes = [
    { id: "showCombinatorNames", label: "Show Combinator Names", variable: "showCombinatorNames", initiallyChecked: true },
    { id: "showValues", label: "Show Values", variable: "showValues", initiallyChecked: true },
    { id: "showParameters", label: "Show Parameters", variable: "showParameters", initiallyChecked: true },
    { id: "hideParameters", label: "Hide Parameter Nodes", variable: "hideParameters", initiallyChecked: false }
];

checkboxes.forEach((checkbox) => {
    const checkboxGroup = checkboxContainer.append("label")
        .style("margin-right", "20px")
        .style("cursor", "pointer");
    
    const input = checkboxGroup.append("input")
        .attr("type", "checkbox")
        .attr("id", checkbox.id)
        .property("checked", checkbox.initiallyChecked);
    
    input.on("change", function() {
        checkboxState[checkbox.variable] = this.checked;
        console.log(checkbox.variable + " = " + checkboxState[checkbox.variable]);
        loadResult(selectedResult);
    });
    
    checkboxGroup.append("text")
        .text(checkbox.label)
        .style("margin-left", "5px");
});

function refreshSelectedResult(change) {
    selectedResult += change;
    const successful = loadResult(selectedResult);
    if (successful) {
        selectedResultText.text(selectedResult);
    } else {
        selectedResult -= change;
    }
}

// ============ D3 Setup ============
const tree = d3.layout.tree()
    .size([tree_height, tree_width]);

const diagonal = d3.svg.diagonal()
    .projection((d) => [d.y, d.x]);

const realSvg = d3.select("body").append("svg")
    .attr("width", canvas_width)
    .attr("height", canvas_height);

const svgDefs = realSvg.append("defs");
const svg = realSvg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Load visualization data
d3.json("./results.json", (error, data) => {
    treeData = data;
    loadResult(selectedResult);
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

function loadResult(resultNum) {
    let resultData = JSON.parse(JSON.stringify(treeData[resultNum]));
    if (resultData === undefined) {
        return false;
    }
    
    root = filterParameterNodes(resultData);
    if (root === null) {
        return false;
    }
    
    root.x0 = tree_height / 2;
    root.y0 = 0;

    // Recursively collapse all children initially
    function collapse(d) {
        if (d.children) {
            d._children = d.children;
            d._children.forEach(collapse);
            d.children = null;
        }
    }
    
    if (root.children) {
        root.children.forEach(collapse);
    }
    
    update(root);
    return true;
}

// ============ Stripe Gradient Management ============
function clearStripes() {
    allStripeGradients = [];
}

function makeStripes(id, colors) {
    const step = 1 / colors.length;
    let offset = 0;

    const grad = svgDefs.append("linearGradient")
        .attr("id", id)
        .attr("spreadMethod", "repeat")
        .attr("x2", "30")
        .attr("gradientUnits", "userSpaceOnUse")
        .attr("gradientTransform", "rotate(-45)");
    
    colors.forEach((color) => {
        grad.append("stop")
            .attr("offset", offset)
            .attr("stop-color", color);
        offset += step;
        grad.append("stop")
            .attr("offset", offset)
            .attr("stop-color", color);
    });
    
    allStripeGradients.push(grad);
}

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

    // Normalize for fixed-depth
    nodes.forEach((d) => { d.y = d.depth * tree_level_depth; });

    // Set unique ID for each node
    const node = svg.selectAll("g.node")
        .data(nodes, (d) => d.id || (d.id = ++nodeIdCounter));

    // Enter any new nodes at the parent's previous position
    const newNodes = node.enter().append("g")
        .attr("class", "node")
        .attr("transform", (d) => "translate(" + source.y0 + "," + source.x0 + ")")
        .on("click", click)
        .on("mouseover", function() {
            // Emphasize labels on hovered node
            d3.select(this).selectAll("text")
                .style("font-size", "16px")
                .style("font-weight", "bold");
            
            // Hide labels on all other nodes
            svg.selectAll("g.node:not(:hover)").selectAll("text")
                .style("fill-opacity", 1e-6);
        })
        .on("mouseout", function() {
            // Restore labels based on checkbox state
            svg.selectAll("g.node").selectAll("text")
                .style("font-size", "12px")
                .style("font-weight", "normal");
            
            svg.select("g.node:hover").selectAll("text")
                .style("fill-opacity", (d, i) => {
                    if (i === 0) return checkboxState.showValues ? 1 : 1e-6;
                    if (i === 1) return checkboxState.showParameters ? 1 : 1e-6;
                    if (i === 2) return checkboxState.showCombinatorNames ? 1 : 1e-6;
                    return 1e-6;
                });
            
            // Restore visibility for non-hovered nodes based on checkbox state
            svg.selectAll("g.node").each(function(d) {
                if (this !== d3.event.target && d3.event.target !== d3.event.relatedTarget) {
                    d3.select(this).select("text.value-text")
                        .style("fill-opacity", checkboxState.showValues ? 1 : 1e-6);
                    d3.select(this).select("text.parameter-text")
                        .style("fill-opacity", checkboxState.showParameters ? 1 : 1e-6);
                    d3.select(this).select("text.combinator-text")
                        .style("fill-opacity", checkboxState.showCombinatorNames ? 1 : 1e-6);
                }
            });
        });

    // Add circle for combinators or rect for parameters
    newNodes.each(function(d) {
        const nodeGroup = d3.select(this);
        if (d.is_combinator) {
            nodeGroup.append("circle")
                .attr("r", 1e-5)
                .style("fill", () => {
                    stripeIdCounter += 1;
                    const stripeName = "stripe" + d.id + stripeIdCounter;
                    makeStripes(stripeName, d.colors);
                    return "url(#" + stripeName + ")";
                });
        } else {
            nodeGroup.append("rect")
                .attr("x", -0.75 * circle_radius)
                .attr("y", -0.75 * circle_radius)
                .attr("width", 1.5 * circle_radius)
                .attr("height", 1.5 * circle_radius)
                .style("fill", () => {
                    stripeIdCounter += 1;
                    const stripeName = "stripe" + d.id + stripeIdCounter;
                    makeStripes(stripeName, d.colors);
                    return "url(#" + stripeName + ")";
                });
        }
    });

    // Add value label (left of node)
    newNodes.append("text")
        .attr("class", "value-text")
        .attr("x", -1.1 * circle_radius)
        .attr("dy", ".35em")
        .attr("text-anchor", "end")
        .text((d) => d.val)
        .style("fill-opacity", 1e-6);
    
    // Add parameter label (above node)
    newNodes.append("text")
        .attr("class", "parameter-text")
        .attr("x", 0)
        .attr("dy", -1.1 * circle_radius)
        .attr("text-anchor", "middle")
        .text((d) => d.parameter)
        .style("fill-opacity", 1e-6)
        .style("text-anchor", "middle");
    
    // Add combinator label (below node)
    newNodes.append("text")
        .attr("class", "combinator-text")
        .attr("x", 0)
        .attr("dy", 1.3 * circle_radius)
        .attr("text-anchor", "middle")
        .text((d) => d.combinator)
        .style("fill-opacity", 1e-6)
        .style("text-anchor", "middle");

    // Reset stripe counter and add default stripes
    makeStripes("myStripes", ["green", "red"]);
    stripeIdCounter = 0;
    
    // Transition nodes to their new position
    const movedNode = node.transition().duration(transition_duration)
        .attr("transform", (d) => "translate(" + d.y + "," + d.x + ")");
    
    clearStripes();
    
    movedNode.select("circle")
        .attr("r", circle_radius);
    
    movedNode.select("rect")
        .attr("x", -0.75 * circle_radius)
        .attr("y", -0.75 * circle_radius)
        .attr("width", 1.5 * circle_radius)
        .attr("height", 1.5 * circle_radius);
    
    movedNode.select("text.value-text")
        .style("fill-opacity", checkboxState.showValues ? 1 : 1e-6);
    
    movedNode.select("text.parameter-text")
        .style("fill-opacity", checkboxState.showParameters ? 1 : 1e-6);
    
    movedNode.select("text.combinator-text")
        .style("fill-opacity", checkboxState.showCombinatorNames ? 1 : 1e-6);

    // Transition exiting nodes to the parent's new position
    const hiddenNodes = node.exit().transition().duration(transition_duration)
        .attr("transform", (d) => "translate(" + source.y + "," + source.x + ")")
        .remove();
    
    hiddenNodes.select("circle")
        .attr("r", 1e-5);
    
    hiddenNodes.select("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", 0)
        .attr("height", 0);
    
    hiddenNodes.selectAll("text")
        .style("fill-opacity", 1e-6);

    // Update the links
    const link = svg.selectAll("path.link")
        .data(links, (d) => d.target.id);

    // Enter any new links at the parent's previous position
    link.enter().insert("path", "g")
        .attr("class", "link")
        .attr("d", (d) => {
            const o = { x: source.x0, y: source.y0 };
            return diagonal({ source: o, target: o });
        })
        .append("svg:title")
            .text((d) => d.target.edge_name);

    // Transition links to their new position
    link.transition().duration(transition_duration)
        .attr("d", diagonal);

    // Transition exiting links to the parent's new position
    link.exit().transition().duration(transition_duration)
        .attr("d", (d) => {
            const o = { x: source.x, y: source.y };
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
    if (d.children) {
        d._children = d.children;
        d.children = null;
    } else {
        d.children = d._children;
        d._children = null;
    }
    update(d);
}