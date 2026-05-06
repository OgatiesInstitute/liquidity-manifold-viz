
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
camera.position.set(0, 60, 100);
new OrbitControls(camera, renderer.domElement);

const matBid = new THREE.MeshPhongMaterial({ color: 0x1155cc, wireframe: false, opacity: 0.8, transparent: true });
const matAsk = new THREE.MeshPhongMaterial({ color: 0xcc2211, wireframe: false, opacity: 0.8, transparent: true });

function buildTerrain(matrix, mat) {
    const rows = matrix.length, cols = matrix[0].length;
    const geo  = new THREE.PlaneGeometry(80, 40, rows-1, cols-1);
    geo.rotateX(-Math.PI / 2);
    const pos = geo.attributes.position;
    matrix.forEach((row, i) =>
        row.forEach((v, j) => pos.setY(i * cols + j, v * 8))
    );
    geo.computeVertexNormals();
    return new THREE.Mesh(geo, mat);
}

let bidMesh = buildTerrain(Array(100).fill(Array(50).fill(0)), matBid);
let askMesh = buildTerrain(Array(100).fill(Array(50).fill(0)), matAsk);
scene.add(bidMesh, askMesh);
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
scene.add(new THREE.DirectionalLight(0xffffff, 0.8));

const faultGeo = new THREE.PlaneGeometry(80, 60);
faultGeo.rotateX(-Math.PI / 2);
const faultLine = new THREE.Mesh(faultGeo, new THREE.MeshBasicMaterial({ color: 0xff0000, opacity: 0.15, transparent: true }));
faultLine.visible = false;
scene.add(faultLine);

const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = ({ data }) => {
    const { bids, asks, fault_score, fault_level } = JSON.parse(data);
    scene.remove(bidMesh, askMesh);
    bidMesh = buildTerrain(bids, matBid);
    askMesh = buildTerrain(asks, matAsk);
    scene.add(bidMesh, askMesh);
    faultLine.visible = fault_score > 0.5;
    if (faultLine.visible) faultLine.position.set(0, 0, fault_level * 40 - 20);
};

(function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
})();
