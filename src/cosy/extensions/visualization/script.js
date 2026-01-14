var margin = {top: 20, right: 120, bottom: 20, left: 120},
    canvas_height = 700,
    canvas_width = 960
    tree_width = canvas_width - margin.right - margin.left,
    tree_height = canvas_height - margin.top - margin.bottom,
    tree_level_depth = 180;

var i = 0,
    duration = 750,
    the_tree_data,
    root;

let selectedResult = 0;

const resultSelector = d3.select("body")
    .append("div");

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

// Function to update counter display
function refreshSelectedResult(change) {
    selectedResult += change
    let successful = loadResult(selectedResult)
    if (successful) {
        selectedResultText.text(selectedResult);
    } else {
        console.log("no success!")
        selectedResult -= change
    }
}

var tree = d3.layout.tree()
    .size([tree_height, tree_width]);

var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

var real_svg = d3.select("body").append("svg")
   .attr("width", canvas_width)
   .attr("height", canvas_height)

var svg_defs = real_svg.append("defs")
var svg = real_svg.append("g")
   .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.json("./results.json", function(error, tree_data) {
    console.log(tree_data)
    the_tree_data = tree_data
    // if (!(tree_data.hasOwnProperty(resultNum))) {
    //     console.log("not there")
    //     return false;
    // }
    loadResult(selectedResult)
});

function loadResult(resultNum) {
    console.log(the_tree_data)
    // if (!(tree_data.hasOwnProperty(resultNum))) {
    //     console.log("not there")
    //     return false;
    // }
    console.log(resultNum)
    root = the_tree_data[resultNum];
    if (root === undefined) {
        console.log("not there2")
        return false;
    }
    console.log(root)
    root.x0 = tree_height / 2;
    root.y0 = 0;

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
    return true
}



all_stripe_grads = []
function clearStripes(){
    all_stripe_grads = []
}

function makeStripes(id, colors){
    var stops = []
    var step = 1 / colors.length
    var offset = 0

    //stops.forEach(function(color) {
    //    stops.push({offset: offset, color: color})
    //    offset += step
    //    stops.push({offset: offset, color: color})
    //})
    //svg_defs.selectAll("linearGradient").remove()
    var grad = svg_defs.append("linearGradient")
        .attr('id',id)
        .attr('spreadMethod',"repeat")
        .attr('x2',"30")
        .attr('gradientUnits',"userSpaceOnUse")
        .attr('gradientTransform',"rotate(-45)")
    colors.forEach(function(color) {
        grad.append("stop")
            .attr("offset", offset)
            .attr("stop-color", color)
        offset += step
        grad.append("stop")
            .attr("offset", offset)
            .attr("stop-color", color)
    })
    all_stripe_grads.push(grad)
}

//var rect = svg.select("rect").attr("fill", "url(#TestGradient)")
$('svg rect').attr('fill','url(#TestGradient)');

d3.select(self.frameElement).style("height", "800px");

const circle_radius = 25;

function update(source) {

  // Compute the new tree layout.
  var nodes = tree.nodes(root),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * tree_level_depth; });

  // Set unique ID for each node
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var new_nodes = node.enter().append("g")
      .attr("class", "node")
      .attr("transform", function(d) { return "translate(" + source.y0 + "," + source.x0 + ")"; })
      .on("click", click);

  new_nodes.append("circle")
      .attr("r", 1e-5)
      .style("fill", function(d) {
        stripe_id += 1
        stripe_name = "stripe" + d.id + stripe_id
        makeStripes(stripe_name, d.colors)
        return "url(#" + stripe_name + ")";
      });
      //.style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });

  new_nodes.append("text")
      .attr("x", -1.1*circle_radius) // function(d) { return d.children || d._children ? -10 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", "end") //function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.val; })
      .style("fill-opacity", 1e-6);
  new_nodes.append("text")
      .attr("x", 0) // function(d) { return d.children || d._children ? -10 : 10; })
      .attr("dy", -1.1*circle_radius)
      .attr("text-anchor", "end") //function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.parameter; })
      .style("fill-opacity", 1e-6)
      .style("text-anchor", "middle");
  new_nodes.append("text")
      .attr("x", 0) // function(d) { return d.children || d._children ? -10 : 10; })
      .attr("dy", 1.3*circle_radius)
      .attr("text-anchor", "end") //function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { console.log(d.combinator); return d.combinator; })
      .style("fill-opacity", 1e-6)
      .style("text-anchor", "middle");


  makeStripes("myStripes", ["green", "red"])
  var stripe_id = 0
  // Transition nodes to their new position.
  var moved_node = node.transition().duration(duration)
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });
  clearStripes()
  moved_node.select("circle")
      .attr("r", circle_radius)
  moved_node.selectAll("text")
      .style("fill-opacity", 1);


  // Transition exiting nodes to the parent's new position.
  var hidden_nodes = node.exit().transition().duration(duration)
      .attr("transform", function(d) { return "translate(" + source.y + "," + source.x + ")"; })
      .remove();
  hidden_nodes.select("circle")
      .attr("r", 1e-6);
  hidden_nodes.select("text2")
      .style("fill-opacity", 1e-6);


  // Update the links…
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; });


  // Enter any new links at the parent's previous position.
  link.enter().insert("path", "g")
      .attr("class", "link")
      .attr("d", function(d) {
        var o = {x: source.x0, y: source.y0};
        return diagonal({source: o, target: o});
      })
      .append("svg:title")
          .text(function(d, i) { return d.target.edge_name; });



  //Transition links to their new position.
  link.transition().duration(duration)
      .attr("d", diagonal);

  // Transition exiting nodes to the parent's new position.
  link.exit().transition().duration(duration)
      .attr("d", function(d) {
        var o = {x: source.x, y: source.y};
        return diagonal({source: o, target: o});
      })
      .remove();


  // Stash the old positions for transition.
  nodes.forEach(function(d) {
    d.x0 = d.x;
    d.y0 = d.y;
  });
}

// Toggle children on click.
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