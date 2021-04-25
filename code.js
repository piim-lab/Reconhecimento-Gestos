
let trainingData = [];

let loadedLabels = []

let labels = [
            "Plano De Fundo"
        ];

let gestureNumber = 1;

let warmedUp = false

//tf.enableDebugMode()
tf.enableProdMode()

function setText( text ) {
    document.getElementById( "status" ).innerText = text;
}

function warmupModel(){
    const warmupResult = model.predict(tf.zeros([ 1, 224, 224, 3 ]));
    warmupResult.dataSync();
    warmupResult.dispose();
}

async function callPredictImage(){
    tf.engine().startScope();
    await predictImage();
    tf.engine().endScope()
}

async function predictImage() {
    if( !hasTrained ) { return; }
    if( !warmedUp ){ warmupModel(); warmedUp = true; }
    const img = await getWebcamImage();
    let result = tf.tidy( () => {
        const input = img.reshape( [ 1, 224, 224, 3 ] );
        return model.predict( input );
    });
    img.dispose();
    let prediction = await result.data();
    result.dispose();
    let id = prediction.indexOf( Math.max( ...prediction ) );
    setText( labels[ id ] +' - ' + (Math.max( ...prediction)*100).toFixed(1) + '%');
}

function createTransferModel( model ) {
    const bottleneck = model.getLayer( "dropout" ); 
    const baseModel = tf.model({
        inputs: model.inputs,
        outputs: bottleneck.output
    });    
    for( const layer of baseModel.layers ) {
        layer.trainable = false;
    }
    const newHead = tf.sequential();
    newHead.add( tf.layers.flatten( {
        inputShape: baseModel.outputs[ 0 ].shape.slice( 1 )
    } ) );
    newHead.add( tf.layers.dense( { units: 100, activation: 'relu' } ) );
    newHead.add( tf.layers.dense( { units: 70, activation: 'relu' } ) );
    newHead.add( tf.layers.dense( {
        units: labels.length,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
    } ) );
    
    const newOutput = newHead.apply( baseModel.outputs[ 0 ] );
    const newModel = tf.model( { inputs: baseModel.inputs, outputs: newOutput } );
    return newModel;
}

const mobilenet = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json";

let model = null;
let hasTrained = false;

async function train(){

    setText("Preparando para treino...");

    trainModel();

    tf.disposeVariables;
}

async function trainModel() {
    
        model = await createTransferModel( model ); 

        hasTrained = false;

        setText( "Treinando..." );

        // Setup training data
        const imageSamples = [];
        const targetSamples = [];
        trainingData.forEach( sample => {
            imageSamples.push( sample.image );
            let cat = [];
            for( let c = 0; c < labels.length; c++ ) {
                cat.push( c === sample.category ? 1 : 0 );
            }
            targetSamples.push( tf.tensor1d( cat ) );
        });
        const xs = tf.stack( imageSamples );
        const ys = tf.stack( targetSamples );

        model.compile( { loss: "meanSquaredError", optimizer: "adam", metrics: [ "acc" ] } );

        var start = window.performance.now();
        tf.setBackend(tf.getBackend()) ;
        await model.fit( xs, ys, {
            epochs: 30,
            shuffle: true,
            batchSize: ~~(trainingData.length/labels.length),
            callbacks: {onEpochEnd: ( epoch, logs ) => {
                            setText("Época nº" + (epoch));
                            console.log(tf.memory());
                            console.log( "Epoch #", epoch, logs );
                        }}/*tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss', 'mse', 'acc'],
                    { height: 200, callbacks: ['onEpochEnd'] }),*/
                    
            
        });
        var end = window.performance.now();
        xs.dispose();
        ys.dispose();
        console.log("Tempo de execução: " + (end - start));
        hasTrained = true;
    
}

async function setupWebcam() {

    return new Promise( ( resolve, reject ) => {
        const webcamElement = document.getElementById( "webcam" );
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
        if( navigator.getUserMedia ) {
            navigator.getUserMedia( { video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener( "loadeddata", resolve, false );
                },
            error => reject());
        }
        else {
            reject();
        }
    });
}

async function getWebcamImage() {

    const img = tf.image.resizeBilinear((( cropImage(await webcam.capture()) )), [224,224]).toFloat();
    const normalized = img.div( tf.scalar(127) ).sub( tf.scalar(1) );
    return normalized;
}


async function captureSample( category ) {

    image = await getWebcamImage()
    
    trainingData.push( {
        image: image,
        category: category
    });
    setText( "Gesto capturado: " + labels[ category ] );
}

let webcam = null;

(async () => {
    textArea = document.getElementById('textArea')
    textArea.value = ''
    model = await tf.loadLayersModel( mobilenet );
    setText('Mobilenetv1 Carregado');
    await setupWebcam();
    webcam = await tf.data.webcam( document.getElementById( "webcam" ) );
    setInterval( callPredictImage, 250 );
})();

function createNewGesture(){
    if (labels.length < 30) {
        textArea = document.getElementById('textArea')
        gestureName = ((textArea.value === undefined || textArea.value.match(/^ *$/) !== null) ? "Gesto "+gestureNumber : textArea.value);
        textArea.value = ''
        addGesture(gestureName);
        labels.push(gestureName);        
    }
}

function addGesture(name){
    gestureNumber++;
    gestureButton = document.getElementById('gestureButton');
    createGestureAddExample(name);
    let whitespace = document.createTextNode("\u00A0"); 
    gestureButton.appendChild(whitespace);

}

function createGestureAddExample(name){
    gesture = document.createElement("button");
    gesture.setAttribute("id", name);''
    gesture.onclick = function() {captureSample(labels.indexOf(event.srcElement.innerText));};
    gesture.innerHTML = name;
    gestureButton = document.getElementById('gestureButton');
    gestureButton.appendChild(gesture);
}

async function saveModel(){
    if(hasTrained){
        await model.save('downloads://my-model');  
        data = labels.toString();
        saveGestures();
        fecharOverlay();
    }
}

function saveGestures(){
    download(data, 'gestures.txt', 'text/plain');
    fecharOverlay();
}

function removeAllGestures(){
    gestureButton = document.getElementById('gestureButton');
    while (gestureButton.firstChild) {
        gestureButton.removeChild(gestureButton.lastChild);
    }
}

function loadGestures(){
    if (loadedLabels.length == 0) {return;}
        removeAllGestures()
        labels = loadedLabels;
        labels.forEach(label => {
            addGesture(label);
        })
        fecharOverlay();
}

async function loadModel(){
    if(!hasTrained){
        loadGestures();
        jsonUpload = document.getElementById('json');
        weightsUpload = document.getElementById('weights');
        jsonFile = new File([jsonUpload.files[0]], 'my-model.json');
        weightsFile = new File([weightsUpload.files[0]], 'my-model.weights.bin');
        model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, weightsFile]));
        hasTrained = 1;
        fecharOverlay();
    }
}

function loadGesturesFile(event){
    gestureUpload = document.getElementById('gestures');
    var fr = new FileReader(); 
    fr.onload=function(){ 
        loadedLabels=[]
        loadedLabels=fr.result.split(',');
        fr.result = ''
        if(loadedLabels[0] != [] || loadedLabels[0] != ''){
            gestures_label = document.getElementsByClassName('gestures')[0];
            gestures_label.style.backgroundColor = "lime";
            gestures_label.textContent = "Arquivo de Gestos Enviado com Sucesso";
            gestures_label.style.color = "black";
        }
    } 
    fr.readAsText(event.currentTarget.files[0]);
}

function download(data, filename, type){
    var file = new Blob([data], {type: type});
    if (window.navigator.msSaveOrOpenBlob)
        window.navigator.msSaveOrOpenBlob(file, filename);
    else {
        var a = document.createElement("a"),
        url = URL.createObjectURL(file);
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        setTimeout(function() {
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);  
        }, 0); 
    }
}

function abrirOverlaySalvar(){
    overlay = document.getElementById('overlay');
    overlay.style.display = "block";
    salvar_div = document.getElementById('salvar_div');
    salvar_div.style.display = "flex";
}

function abrirOverlayCarregar(){
    overlay = document.getElementById('overlay');
    overlay.style.display = "block";
    carregar_div = document.getElementById('carregar_div');
    carregar_div.style.display = "block";
}

function fecharOverlay(){
    overlay = document.getElementById('overlay');
    overlay.style.display = "none";
    carregar_div = document.getElementById('salvar_div');
    salvar_div.style.display = "none";
    carregar_div = document.getElementById('carregar_div');
    carregar_div.style.display = "none";
}

function alterarLabel(label){
    if(label === 'json'){
        json_label = document.getElementsByClassName('json')[0];
        json_label.style.backgroundColor = "lime";
        json_label.textContent = "Arquivo JSON Enviado com Sucesso";
        json_label.style.color = "black";

    } else{
        if (label === 'weights'){
            weights_div = document.getElementsByClassName('weights')[0];
            weights_div.style.backgroundColor = "lime";
            weights_div.textContent = "Arquivo Binário de Pesos Enviado com Sucesso";
            weights_div.style.color = "black";
        }
    }
}
function cropImage(img) {  
    const width = img.shape[0];  const height = img.shape[1];  // use the shorter side as the size to which we will crop  
    const shorterSide = Math.min(img.shape[0], img.shape[1]);  // calculate beginning and ending crop points 
    const startingHeight = (height - shorterSide) / 2;  
    const startingWidth = (width - shorterSide) / 2;  
    const endingHeight = startingHeight + shorterSide;  
    const endingWidth = startingWidth + shorterSide;  // return image data cropped to those points
    return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
}