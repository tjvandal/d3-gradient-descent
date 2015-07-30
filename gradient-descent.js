/**
 * gradient-descent.js
 * Visualizes the gradient descent algorithm for linear regression. 
 * PGA tour distance predicting accuracy
 * @author: TJ Vandal,  Modified from Hugo Janssen
 * @date:   7/30/2015
 */
"use strict";

function gradientDescent(elt, w, h, numPoints, learningRate) {

	var iter = 1,
		points = [],
		costs = [],
		theta0 = -2,
		theta1 = 2,
		alpha = 1.95,
		convergenceTh = 0.0001,
		isConverged = false,
		maxIter = 1000,
	    i=0,
	    mse_before = 0,
	    prevtheta0 = 0;
	
	var numberFormat = d3.format(".4f");
	
	var margin = {top: 25, right: 25, bottom: 50, left: 50},
		width = w - margin.left - margin.right,
		height = h - margin.top - margin.bottom;
	
	var svg = d3.select(elt).append("svg")
		.style("width", width + margin.left + margin.right)
		.style("height", height + margin.top + margin.bottom);
	
	
	// The hypothesis plot
	var xHypothesis = d3.scale.linear()
		.domain([-5, 5])
		.range([0, w / 3 - margin.left - margin.right]);
	
	var yHypothesis = d3.scale.linear()
		.domain([-5, 5])
		.range([height, 0]);
		
	var xHypothesisAxis = d3.svg.axis()
		.scale(xHypothesis)
		.orient("bottom");
		
	var yHypothesisAxis = d3.svg.axis()
		.scale(yHypothesis)
		.orient("left");

	var hypothesisGroup = svg.append("g")
		.attr("class", "hypothesis")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
		
	hypothesisGroup.append("g").append("text")
		.attr("x", width / 6)
		.attr("class", "title")
		.style("text-anchor", "middle")
		.text("Training data and regression line");
		
	hypothesisGroup.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + height + ")")
	.call(xHypothesisAxis)
		.append("text")
		.attr("x", width / 3 - margin.right)
		.attr("y", -12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("Distance (normalized)");


	hypothesisGroup.append("g")
		.attr("class", "y axis")
		.call(yHypothesisAxis)
		.append("text")
		.attr("transform", "rotate(-90)")
		.attr("y", 12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("Accuracy (normalized)");

	hypothesisGroup.append("g")
		.append("text")
		.attr("x", width / 6)
		.attr("y", margin.top)
		.attr("class", "function-label")
		.style("text-anchor", "middle")
		.text("");
		
		
	// The cost function plot
	var xCostFunction = d3.scale.linear()
		.domain([-2, 2])	
		.range([0, w / 3 - margin.left - margin.right]);
	
	var yCostFunction = d3.scale.linear()
		.domain([-2, 2])
		.range([height, 0]);
		
	var xCostFunctionAxis = d3.svg.axis()
		.scale(xCostFunction)
		.orient("bottom");
		
	var yCostFunctionAxis = d3.svg.axis()
		.scale(yCostFunction)
		.orient("left");

	var costFunctionGroup = svg.append("g")
		.attr("class", "cost-function")
		.attr("transform", "translate(" + (2 * margin.left + width/ 3) + "," + margin.top + ")");
		
	costFunctionGroup.append("g").append("text")
		.attr("x", width / 6)
		.attr("class", "title")
		.style("text-anchor", "middle")
		.text("Parameters");
		
	costFunctionGroup.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + height + ")")
		.call(xCostFunctionAxis)
	.append("text")
		.attr("x", width / 3 - margin.right)
		.attr("y", -12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("θ₁ (slope)");

	costFunctionGroup.append("g")
		.attr("class", "y axis")
		.call(yCostFunctionAxis)
	.append("text")
		.attr("transform", "rotate(-90)")
		.attr("y", 12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("θ₀ (intercept)");

	var JFunctionGroup = svg.append("g")
		.attr("class", "jfunction")
		.attr("transform", "translate(" + (2 * margin.left + width * 2 / 3) + "," + margin.top + ")");

	// The cost function plot
	var xJFunction = d3.scale.linear()
		.domain([-2, 2])	
		.range([0, w / 3 - margin.left - margin.right]);
	
	var yJFunction = d3.scale.linear()
		.domain([0, 6])
		.range([height, 0]);
		
	var xJFunctionAxis = d3.svg.axis()
		.scale(xJFunction)
		.orient("bottom");
		
	var yJFunctionAxis = d3.svg.axis()
		.scale(yJFunction)
		.orient("left");

	JFunctionGroup.append("g").append("text")
		.attr("x", width / 6)
		.attr("class", "title")
		.style("text-anchor", "middle")
		.text("Cost Function");
		
	JFunctionGroup.append("g")
		.attr("class", "x axis")
		.attr("transform", "translate(0," + height + ")")
	.call(xJFunctionAxis)
		.append("text")
		.attr("x", width / 3 - margin.right)
		.attr("y", -12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("θ₀ (intercept)");

	JFunctionGroup.append("g")
		.attr("class", "y axis")
	.call(yJFunctionAxis)
		.append("text")
		.attr("transform", "rotate(-90)")
		.attr("y", 12)
		.attr("dy", ".35em")
		.style("text-anchor", "end")
		.text("J(θ₀,θ₁)");




	// Text label
	svg.append("g")
		.append("text")
		.attr("x", margin.left)
		.attr("y", height + margin.top + margin.bottom / 2)
		.attr("dy", ".35em")
		.attr("class", "status-label")
		.style("text-anchor", "center")
		.text("");
	
	
	/** 
	 * Returns the regression function
	 */ 
	function hypothesis(x) {
		return theta1 * x + theta0;
	}
	
	/**
	 * Returns the difference between the predicted value and the actual value 
	 * for a specific point.
	 */
	function predictionError(point) {
		return hypothesis(point.x) - point.y;
	}
	
	/**
	 * Returns the squared error for a specific point.
	 */
	function squaredError(point) {	
		return Math.pow(predictionError(point), 2);
	}
	
	/**
	 * Returns the mean squared error for all points in the dataset.
	 */
	function meanSquaredError(data) {
		var sum = 0;
		data.forEach(function(d) {
			sum += squaredError(d);
		});
		return sum / (2 * data.length);
	}
	
	/**
	 * The partial derivative of the cost function for theta0.
	 */
	function derivativeTheta0(points) {
		var sum = 0;
		points.forEach(function(d) {
			sum += predictionError(d);
		});
		return sum / points.length;		
	}
	
	/**
	 * The partial derivative of the cost function for theta1.
	 */
	function derivativeTheta1(points) {
		var sum = 0;
		points.forEach(function(d) {
			sum += predictionError(d) * d.x;
		});
		return sum / points.length;
	}
	
	/**
	 * Appends the data points to the plot.
	 */
	function appendPoints(points) {

		hypothesisGroup.selectAll(".circle")
				.data(points)
			.enter().append("circle")
				.attr("id", function(d) { return d.id; })
				.attr("cx", function(d) { return xHypothesis(d.x); })
				.attr("cy", function(d) { return yHypothesis(d.y); })
				.attr("r", 4);   
	}

	/**
	 * Updates the chart.
	 */
	function update(points) {
		var x1 = xHypothesis.domain()[0];
		var x2 = xHypothesis.domain()[1];
		
		var y1 = hypothesis(x1);
		var y2 = hypothesis(x2);
		
		// Draw and update the regression line
		var line = hypothesisGroup.selectAll(".regression-line")
			.data([{y1, y2}]);
		
		line.enter().append("line")
			.attr("class", "regression-line")
			.attr("x1", function(d) { return xHypothesis(x1); })
			.attr("x2", function(d) { return xHypothesis(x2); });
		
		line.transition().delay(0).duration(500)
			.attr("y1", function(d) { return yHypothesis(d.y1); })
			.attr("y2", function(d) { return yHypothesis(d.y2); });

		// Draw the cost function circles
		var circle = costFunctionGroup.selectAll(".circle")
			.data([{theta0, theta1}])
		.enter().append("circle")
			.attr("cx", function(d) { return xCostFunction(d.theta0); })
			.attr("cy", function(d) { return yCostFunction(d.theta1); })
			.attr("r", 2);        
			
		// Costing stuff
		var mse_now = meanSquaredError(points);
		var circle = JFunctionGroup.selectAll(".circle")
			.data([{theta0, mse_now}])
		.enter().append("circle")
			.attr("cx", function(d) { return xJFunction(d.theta0); })
			.attr("cy", function(d) { return yJFunction(d.mse_now); })
			.attr("r", 2); 

		if(mse_before > 0){
			var costline = JFunctionGroup.selectAll(".cost-line")
				.data([{prevtheta0, theta0, mse_before, mse_now}])
			.enter().append("line")
				.attr("x1", function(d){return xJFunction(d.prevtheta0)})
				.attr("x2", function(d){return xJFunction(d.theta0)})
				.attr("y1", function(d){return yJFunction(d.mse_before)})
				.attr("y2", function(d){return yJFunction(d.mse_now)})
				.attr("stroke",  "#33f")
				.attr("stroke-width", "1.5px");
        }

		// Update the labels
		svg.selectAll(".status-label").text("Iteration " + iter + 
			"; learningRate=" + alpha + "; convergence=" + convergenceTh + 
			"; mse=" + numberFormat(meanSquaredError(points)));
		
		svg.selectAll(".function-label").text("hθ(x) = " + 
			numberFormat(theta1) + " • x + "  + numberFormat(theta0));
	}
	
	/**
	 * Executes one iteration of the algorithm
	 */
	function iterate(points) {
		mse_before = meanSquaredError(points);

		// The descent step
		prevtheta0 = theta0;
		var temp0 = theta0 - (alpha * derivativeTheta0(points));
		var temp1 = theta1 - (alpha * derivativeTheta1(points));
		theta0 = temp0;
		theta1 = temp1;
		
		isConverged = (mse_before - meanSquaredError(points) < convergenceTh); 
		// Update the chart
		update(points);
	}

	function getPGA(rows){
		var points = [];
		rows.forEach(function(d) { 
			d.id = pga.length;
			d.accuracy = +d.accuracy;
			d.distance = +d.distance;
			points.push(d);
		});
	}
	
	/** 
	 * The main function initializes the algorithm and calls an iteration every 
	 * 100 milliseconds.
	 */
	function initialize() {
		
		// points = getPGA();

		d3.csv("pga.csv", function(points) {
			points.forEach(function(d) {
				d.id = i; 
				i++;
				d.x = +d.distance;
				d.y = +d.accuracy;
			});

			// normalize points
			var xmean = d3.mean(points, function(d){return d.x});
			var ymean = d3.mean(points, function(d){return d.y});
			var xstd = d3.deviation(points, function(d){return d.x});
			var ystd = d3.deviation(points, function(d){return d.y});
			points.forEach(function(d){
				d.x = (d.x - xmean)/xstd;
				d.y = (d.y - ymean)/ystd;
			});

			// Append points to the chart
			appendPoints(points);
			
			// Initial drawing
			update(points);

			var interval = setInterval(function() {
				if(!isConverged & iter < maxIter) {
					iterate(points);
					iter++;
				} else {
					clearInterval(interval);
				}
			}, 100);
		});
	};
	// Call the main function
	initialize();
}