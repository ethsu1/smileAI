import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import LoadingOverlay from 'react-loading-overlay';
import PacmanLoader from "react-spinners/PacmanLoader";
import {Jumbotron, Row } from 'react-bootstrap'
import { Button, Grid, CardActionArea, Card, CardActions, Container, Typography} from '@material-ui/core';
import './styles.css';
import axios from 'axios';
let canvas,canvasCtx, video, newImage;
class Camera extends React.Component{
	constructor(props){
		super(props);
		this.state ={
			video: null,
			wait: false,
			confidence: "",
		}
		this.sendPhoto = this.sendPhoto.bind(this)
		this.disableButton = this.disableButton.bind(this)
	}
	componentDidMount(){
    	var constraints = { audio: true, video: { width: 500, height: 500}, facingMode: { exact: "user" } };
    	video = document.querySelector("#video");
	    canvas = document.querySelector("#canvas");
		video.width = constraints["video"]["width"];
  		video.height = constraints["video"]["height"];
  		canvas.width = constraints["video"]["width"];
	    canvas.height = constraints["video"]["height"];
	    canvasCtx = canvas.getContext('2d');
	    canvasCtx.translate(canvas.width, 0);
		canvasCtx.scale(-1, 1);
    	navigator.mediaDevices.getUserMedia(constraints)
    	.then((stream) => {
	        video.srcObject = stream;
	        video.muted = true;
	        video.onloadedmetadata = () => {
	        	video.play();
	        };
     	})
     	.catch(function(err) {
        	console.log(err)
        }); 
	}
	async disableButton() {
		if(this.state.wait === false){
			canvasCtx.clearRect(0,0,500,500);
			this.setState({confidence: ""});
			await this.setState({wait: true}, () => this.sendPhoto())
		}
	}
	async sendPhoto(){
		var hidden_canvas = document.createElement('canvas');
		hidden_canvas.width = canvas.width;
		hidden_canvas.height = canvas.height;
		var newCtx = hidden_canvas.getContext('2d');
		newCtx.translate(hidden_canvas.width, 0);
		newCtx.scale(-1, 1);
		var newImage = document.createElement('img');
		newCtx.drawImage(video,0,0,500,500)
		newImage.src = hidden_canvas.toDataURL()
		hidden_canvas.toBlob((blob) => {
		const formData = new FormData();
			formData.append("image", blob);
			axios({
				method: 'post',
    			url: 'http://127.0.0.1:8000/',
    			data: formData,
    			headers: {'Content-Type': 'multipart/form-data' }
			})
			.then(res => {
				let top, left, size, color;
				if(res.data.label === "no face"){
					top = 100;
					left = 100;
					size = 200;
					color = 'red';
				}
				else{
					top = res.data.top;
					left = res.data.left;
					size = res.data.bottom-top;
					if(res.data.probability < 0.65){
						color = 'red';
					}
					else{
						color = 'green';
					}
				}

				canvasCtx.drawImage(newImage,0,0,500,500)
				canvasCtx.save();
				canvasCtx.beginPath();
				canvasCtx.rect(left,top, size, size)
				canvasCtx.strokeStyle = color
				canvasCtx.stroke(); 
				canvasCtx.translate(canvas.width, 0);
				canvasCtx.scale(-1, 1);
				canvasCtx.fillStyle = "white";
				canvasCtx.font = "20pt Arial";
				var prob = res.data.probability*100
				prob =  prob.toFixed(2) + "%"
				var text = res.data.label;
				canvasCtx.fillText(text, 500-size-left,top)
				canvasCtx.restore();
				this.setState({wait: false})
				this.setState({confidence: "Probability of " + text + ": " + prob})
			})
			.catch(err =>{
				//console.log(err)
				})
			})
	}



	render(){
		return(
			<Container>
				<Jumbotron>
						<Row className="justify-content-md-center">
					  		<h1 className="title">smileAI</h1>
					  	</Row>
					  	<Row className="justify-content-md-center">
						  <p>
						    Simple AI that detects smiles.
						  </p>
						</Row>
						<Row className="justify-content-md-center">
							<p>
								<small>Convolutional neural network built using only numpy (no deep learning libaries).</small>
							</p>
						</Row>
						<Row className="justify-content-md-center">
							<p>Code for neural network:&nbsp;</p>
							<a href="https://github.com/ethsu1/convnet_numpy">github.com/ethsu1/convnet_numpy</a>
						</Row>
						<Row className="justify-content-md-center">
							<p>Code for UI:&nbsp;</p>
							<a href="https://github.com/ethsu1/smileAI">github.com/ethsu1/smileAI</a>
						</Row>
					</Jumbotron>
				<Grid container direction="row" justify="center" spacing={5}>
					<Grid item>
							<Card>
								<CardActionArea>
									<video id="video" autoPlay={true} playsInline={true} src={this.state.video} className="video"></video>
								</CardActionArea>
								<CardActions>
									<Button onClick={this.disableButton} variant="contained" color="primary" disabled={this.state.wait}>Take Picture</Button>
								</CardActions>
							</Card>
					</Grid>
					<Grid item>
						<Card>
							<CardActionArea>
								<LoadingOverlay active={this.state.wait} spinner={<PacmanLoader/>}>
									<canvas id="canvas"></canvas>
								</LoadingOverlay>
							</CardActionArea>
						</Card>
							<CardActions>
									<Typography variant="h4">{this.state.confidence}</Typography>
							</CardActions>
					</Grid>

				</Grid>
			</Container>

		)
	}
}

export default Camera;