const zeroPad = (num, places) => String(num).padStart(places, '0')

let preTrainedModel
let trainedModel
let songsdb
let musicContainers = []

// import {faceapi} from './faceapi.min.js'

//classes {0: 'Angry', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprise', 5: 'Neutral'}

const webcam = new Webcam(document.getElementById('webcam'))
const predictionTexts = [
    "Angry", "Happy", "Sad", "Surprised", "Neutral"
]
// .map(e => "You are " + e)

const prediction = document.getElementById('prediction')
const predictionProb = document.getElementById('prediction-prob')
const outputContainer = document.getElementById('output-container')
const songsContainer = document.getElementById('songs-container')
const classes = 5
const numSongs = 5
let isPredicting = false

async function loadModel(){
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json')
    const layer = mobilenet.getLayer('conv_pw_13_relu')
    preTrainedModel = tf.model({inputs: mobilenet.inputs, outputs: layer.output})
    trainedModel = await tf.loadLayersModel('./model/my_model.json')
}



function startPredicting() {
    isPredicting = true
    var id = setInterval(async() => {
        if(isPredicting){
            const predictions = tf.tidy(() => {
                
                const img = webcam.capture()
                const activation = preTrainedModel.predict(img)
                const predictions = trainedModel.predict(activation)
                return predictions.as1D()
            })
            updateBars(await predictions.data())
            updateSongs(await predictions.argMax().squeeze().data())
        } else {
            clearInterval(id)
        }  
    }, 1000)
}

function stopPredicting(){
    isPredicting = false
}


async function init(){
    await webcam.setup()
    await loadModel()
    tf.tidy(() => trainedModel.predict(
        preTrainedModel.predict(webcam.capture())
    ))
    await loadSongs()
    setupProgressBars()
    setUpSongContainers()
}


function setUpSongContainers(){
    for(let i=0; i<5; i++){
        var songContainer = document.createElement("div")
        songContainer.classList.add("individual-song")
        songContainer.innerHTML+=`
            <div class="song-details">
                <h2>
                    <span id="name-song" style="font-weight: normal;"></span> &nbsp; by &nbsp;<span id="artist-song" style="font-weight: normal;"></span>
                </h2>
            </div>
            <div class="play-song">
                <a id="link-song">
                    <img class="play-button" src="./button/play_song.svg" alt="Play-Logo" srcset="">
                </a>
            </div>
        `
        songsContainer.appendChild(songContainer)
        musicContainers.push(songContainer)
    }
    console.log(musicContainers)
}

function setupProgressBars(){
    outputContainer.innerHTML = ""
    for(let i=0; i<classes; i++){
        html = `
        <div class="progress">
            <div class="progress-label">${predictionTexts[i]} : </div>
            <div class="progress-container">
                <div class="progress-value" id="progress_${i}"></div>
            </div>
        </div>
        <br>
        `
        outputContainer.innerHTML += html
    }
}

function updateBars(preds){
    let bars = [...Array(classes).keys()].map(i => document.getElementById(`progress_${i}`))
    for(let i=0; i<classes; i++){
        bars[i].style.width = `${preds[i]*100}%`
    }
}

async function loadSongs(){
    songsdb = await dfd.read_csv("http://"+window.location.host+'/data_emotions.csv')
}

async function updateSongs(classIdx){
    console.log("classIdx", classIdx)
    console.log("predText", predictionTexts[classIdx])
    // document.getElementById('songs-container').innerText = JSON.stringify(
    //     await sampleSongs(predictionTexts[classIdx])
    // )
    // let songsContainer = document.getElementById('songs-container');
    let songsList = await sampleSongs(predictionTexts[classIdx]);
    // songsContainer.innerHTML=``;
    console.log("songsList: ", songsList)
    // songsList.forEach(song => {
    //     songsContainer.innerHTML+=`<div class="individual-song">
    //     <div class="song-details">
    //         <h5>Title : <span id="name-song" style="font-weight: normal;">${song.name}</span></h5>
    //         <h6>Artist : <span id="artist-song" style="font-weight: normal;">${song.artist}</span></h6>
    //     </div>
    //     <div class="play-song">
    //         <a href="${song.id}"><img class="play-button" src="./button/play_song.svg" alt="Play-Logo" srcset=""></a>
    //         <div class="duration-song">${song.length}</div>
    //     </div>
    // </div>`
    // })
    // console.log(songsContainer)
    for(let i=0; i<numSongs; i++){
        var song = songsList[i]
        var songContainer = musicContainers[i]
        console.log(songContainer)
        songContainer.querySelector("#name-song").innerText = song.name
        songContainer.querySelector("#artist-song").innerText = song.artist
        songContainer.querySelector("#link-song").href = song.id
    }
}

async function sampleSongs(mood, n=5, preference = {popularity: 1}){
    var filter = await songsdb.query({column: "mood", is: "==", to: mood})
    var weights = tf.keep(tf.zeros([filter.shape[0]]))
    for(const [key, value] of Object.entries(preference)){
        weights = tf.keep(tf.tidy(()=>{
            return tf.add(
                weights,
                tf.mul(
                    tf.tensor(filter[key].data),
                    tf.scalar(value)
                )
            )
        })) 
    }
    let idxs = []
    for(let i=0; i<n; i++){
        idxs.push((await
            (await
                tf.whereAsync(
                    weights.cumsum().greaterEqual(
                        weights.sum().mul(tf.randomUniform([]))
                    )
                )
            ).slice([0], [1]).squeeze().data()
        )[0])
    }
    var data = filter.iloc({rows: idxs}).loc({columns: ['name', 'album', 'artist', 'id', 'length']})
    var result = JSON.parse(await data.to_json())
    result.map(e => {
        e.id = "https://play.spotify.com/track/"+e.id
        e.length = `${zeroPad((e.length/60000).toFixed(), 2)}:${zeroPad(((e.length%60000)/1000).toFixed(), 2)}`
    })
    return result
}

init()