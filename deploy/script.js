const zeroPad = (num, places) => String(num).padStart(places, '0')

let songsdb
let musicContainers = []
let inputVideo

const predictionTexts = [
    "Angry", "Happy", "Sad", "Surprised", "Neutral"
]


const prediction = document.getElementById('prediction')
const predictionProb = document.getElementById('prediction-prob')
const outputContainer = document.getElementById('output-container')
const songsContainer = document.getElementById('songs-container')
const classes = 5
const numSongs = 5
let isPredicting = false


async function loadModel(){
    Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('/model'),
        faceapi.nets.faceExpressionNet.loadFromUri('/model')
    ])
}

function startVideo() {
    inputVideo = document.getElementById('webcam')
    navigator.getUserMedia(
        {video: {}},
        stream => inputVideo.srcObject = stream,
        err => console.log(err)
    )
}

function startPredicting() {
    isPredicting = true
    var id = setInterval(async() => {
        if(isPredicting){
            const detections = await faceapi.detectAllFaces(
                inputVideo,
                new faceapi.TinyFaceDetectorOptions()  
            ).withFaceExpressions()
            // console.log(detections[0].expressions)
            updateBars(detections[0].expressions)
            updateSongs(await predictions.argMax().squeeze().data()) //TODO
        } else {
            clearInterval(id)
        }  
    }, 1000)
}
window.startPredicting = startPredicting;

function stopPredicting(){
    isPredicting = false
}
window.stopPredicting = stopPredicting;

async function init(){
    // await webcam.setup()
    await loadModel()
    startVideo()
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
        var html = `
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

function updateBars(detections){
    console.log('update-bars', detections)
    //"Angry", "Happy", "Sad", "Surprised", "Neutral"
    let bars = [...Array(classes).keys()].map(i => document.getElementById(`progress_${i}`))
    bars[0].style.width = `${preds[i]*100}%`
    bars[1].style.width = `${preds[i]*100}%`
    bars[2].style.width = `${preds[i]*100}%`
    bars[3].style.width = `${preds[i]*100}%`
    bars[4].style.width = `${preds[i]*100}%`
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

// async function updateBars(preds){
//     var bars = [
//         document.getElementById("neutral_bar"),
//         document.getElementById("happy_bar"),
//         document.getElementById("sad_bar"),
//         document.getElementById("angry_bar"),
//         document.getElementById("fearful_bar"),
//         document.getElementById("disgusted_bar"),
//         document.getElementById("surprised_bar")
//     ]
//     const sw = bars.map(bar => parseFloat(bar.style.width))
//     const fw = [preds.neutral, preds.happy, preds.sad, preds.angry, preds.fearful, preds.disgusted, preds.surprised].map(e => e*100)
//     const nFrames = 100
//     let count = 0

//     var id = setInterval(() => {
//         if (count < nFrames) {
//             for(let i=0; i<bars.length; i++){
//                 bars[i].style.width = (sw[i] + (fw[i]-sw[i])*count/nFrames) + "%"
//             }
//             count++
//         } else {
//             clearInterval(id)
//         }
//     }, 10);
    
// }