import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

globalThis.FileReader = class FileReader {
  async readAsArrayBuffer(blob) {
    this.result = await blob.arrayBuffer();
    this.onloadend?.();
  }
};

const __dirname = dirname(fileURLToPath(import.meta.url));
const outputDir = resolve(__dirname, '../public/models/3d');

const exporter = new GLTFExporter();

const materials = {
  mint: new THREE.MeshStandardMaterial({ color: '#8ce0bd', roughness: 0.48, metalness: 0.02 }),
  mintDark: new THREE.MeshStandardMaterial({ color: '#2f6655', roughness: 0.58, metalness: 0.05 }),
  plastic: new THREE.MeshPhysicalMaterial({ color: '#38cde3', roughness: 0.22, transmission: 0.12, thickness: 0.4, transparent: true, opacity: 0.86 }),
  glass: new THREE.MeshPhysicalMaterial({ color: '#a7f3d0', roughness: 0.04, transmission: 0.46, thickness: 0.8, transparent: true, opacity: 0.52 }),
  paper: new THREE.MeshStandardMaterial({ color: '#f8fafc', roughness: 0.72 }),
  paperGreen: new THREE.MeshStandardMaterial({ color: '#d9f99d', roughness: 0.72 }),
  cardboard: new THREE.MeshStandardMaterial({ color: '#c58d4f', roughness: 0.82 }),
  cardboardEdge: new THREE.MeshStandardMaterial({ color: '#8f5f33', roughness: 0.76 }),
  metal: new THREE.MeshStandardMaterial({ color: '#cbd5e1', roughness: 0.26, metalness: 0.9 }),
  metalDark: new THREE.MeshStandardMaterial({ color: '#64748b', roughness: 0.35, metalness: 0.72 }),
  amber: new THREE.MeshStandardMaterial({ color: '#f4b942', roughness: 0.4 }),
  black: new THREE.MeshStandardMaterial({ color: '#111827', roughness: 0.84 }),
  stone: new THREE.MeshStandardMaterial({ color: '#78716c', roughness: 0.62 }),
  fabric: new THREE.MeshStandardMaterial({ color: '#6366f1', roughness: 0.88 }),
  fabricLight: new THREE.MeshStandardMaterial({ color: '#a5b4fc', roughness: 0.86 }),
  foodGreen: new THREE.MeshStandardMaterial({ color: '#84cc16', roughness: 0.72 }),
  foodOrange: new THREE.MeshStandardMaterial({ color: '#f59e0b', roughness: 0.76 }),
  label: new THREE.MeshStandardMaterial({ color: '#f8fafc', roughness: 0.5 }),
};

function mesh(geometry, material, { position = [0, 0, 0], rotation = [0, 0, 0], scale = [1, 1, 1] } = {}) {
  const object = new THREE.Mesh(geometry, material);
  object.position.set(...position);
  object.rotation.set(...rotation);
  object.scale.set(...scale);
  object.castShadow = true;
  object.receiveShadow = true;
  return object;
}

function makeBase(color = '#20352f') {
  const group = new THREE.Group();
  group.add(mesh(new THREE.CylinderGeometry(1.18, 1.18, 0.12, 56), new THREE.MeshStandardMaterial({ color, roughness: 0.75 }), { position: [0, -1.05, 0] }));
  return group;
}

function makeHandle() {
  return mesh(new THREE.TorusGeometry(0.46, 0.055, 18, 64, Math.PI), materials.label, {
    position: [0, 0.64, 0.03],
    rotation: [0, 0, Math.PI],
    scale: [1.08, 1.36, 1],
  });
}

const builders = {
  plastic() {
    const group = makeBase('#164e63');
    group.add(mesh(new THREE.CylinderGeometry(0.52, 0.64, 1.62, 64), materials.plastic, { position: [0, -0.14, 0] }));
    group.add(mesh(new THREE.CylinderGeometry(0.3, 0.38, 0.42, 64), materials.plastic, { position: [0, 0.88, 0] }));
    group.add(mesh(new THREE.CylinderGeometry(0.32, 0.32, 0.2, 64), materials.mintDark, { position: [0, 1.19, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.84, 0.42, 0.035), materials.label, { position: [0, -0.22, 0.61] }));
    group.add(mesh(new THREE.TorusGeometry(0.42, 0.025, 12, 48), materials.label, { position: [0, -0.78, 0], rotation: [Math.PI / 2, 0, 0] }));
    return group;
  },
  glass() {
    const group = makeBase('#064e3b');
    group.add(mesh(new THREE.CylinderGeometry(0.5, 0.58, 1.65, 64), materials.glass, { position: [0, -0.18, 0] }));
    group.add(mesh(new THREE.CylinderGeometry(0.22, 0.33, 0.62, 64), materials.glass, { position: [0, 0.86, 0] }));
    group.add(mesh(new THREE.TorusGeometry(0.23, 0.04, 16, 48), materials.label, { position: [0, 1.22, 0], rotation: [Math.PI / 2, 0, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.68, 0.34, 0.035), materials.label, { position: [0, -0.16, 0.56] }));
    return group;
  },
  paper() {
    const group = makeBase('#365314');
    for (let i = 0; i < 7; i += 1) {
      group.add(mesh(new THREE.BoxGeometry(1.62, 0.045, 1.06), i % 2 ? materials.paperGreen : materials.paper, {
        position: [0, -0.54 + i * 0.11, 0],
        rotation: [0.015 * i, -0.025 * i, -0.04 * i],
      }));
    }
    for (let i = 0; i < 4; i += 1) {
      group.add(mesh(new THREE.BoxGeometry(1.0, 0.012, 0.02), materials.mintDark, { position: [0.08, -0.12 + i * 0.11, 0.54] }));
    }
    return group;
  },
  cardboard() {
    const group = makeBase('#6b3f1d');
    group.add(mesh(new THREE.BoxGeometry(1.55, 1.08, 1.28), materials.cardboard, { position: [0, -0.34, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.78, 0.1, 1.28), materials.cardboard, { position: [-0.47, 0.3, 0], rotation: [0, 0, 0.38] }));
    group.add(mesh(new THREE.BoxGeometry(0.78, 0.1, 1.28), materials.cardboard, { position: [0.47, 0.3, 0], rotation: [0, 0, -0.38] }));
    group.add(mesh(new THREE.BoxGeometry(0.06, 1.02, 0.04), materials.cardboardEdge, { position: [0, -0.33, 0.66] }));
    group.add(mesh(new THREE.BoxGeometry(1.34, 0.08, 0.04), materials.cardboardEdge, { position: [0, -0.88, 0.66] }));
    return group;
  },
  metal() {
    const group = makeBase('#334155');
    group.add(mesh(new THREE.CylinderGeometry(0.56, 0.56, 1.6, 72), materials.metal, { position: [0, -0.18, 0] }));
    group.add(mesh(new THREE.TorusGeometry(0.44, 0.045, 16, 64), materials.label, { position: [0, 0.66, 0], rotation: [Math.PI / 2, 0, 0] }));
    group.add(mesh(new THREE.TorusGeometry(0.44, 0.045, 16, 64), materials.metalDark, { position: [0, -0.98, 0], rotation: [Math.PI / 2, 0, 0] }));
    group.add(mesh(new THREE.TorusGeometry(0.22, 0.02, 12, 36), materials.metalDark, { position: [0.12, 0.73, 0.1], rotation: [Math.PI / 2, 0.35, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.82, 0.34, 0.035), materials.label, { position: [0, -0.2, 0.57] }));
    return group;
  },
  battery() {
    const group = makeBase('#422006');
    group.add(mesh(new THREE.CylinderGeometry(0.43, 0.43, 1.82, 56), materials.amber, { rotation: [0, 0, Math.PI / 2] }));
    group.add(mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.17, 56), materials.metal, { position: [1.0, 0, 0], rotation: [0, 0, Math.PI / 2] }));
    group.add(mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.15, 56), materials.black, { position: [-1.0, 0, 0], rotation: [0, 0, Math.PI / 2] }));
    group.add(mesh(new THREE.BoxGeometry(0.52, 0.08, 0.06), materials.black, { position: [0.42, 0.46, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.08, 0.52, 0.06), materials.black, { position: [0.42, 0.46, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.44, 0.08, 0.06), materials.black, { position: [-0.42, 0.46, 0] }));
    return group;
  },
  clothes() {
    const group = makeBase('#312e81');
    group.add(mesh(new THREE.BoxGeometry(1.18, 0.88, 0.72), materials.fabric, { position: [0, -0.22, 0] }));
    group.add(mesh(new THREE.BoxGeometry(0.55, 0.42, 0.64), materials.fabricLight, { position: [-0.82, 0.0, 0], rotation: [0, 0, -0.52] }));
    group.add(mesh(new THREE.BoxGeometry(0.55, 0.42, 0.64), materials.fabricLight, { position: [0.82, 0.0, 0], rotation: [0, 0, 0.52] }));
    group.add(mesh(new THREE.TorusGeometry(0.24, 0.035, 14, 42), materials.label, { position: [0, 0.31, 0.37], rotation: [Math.PI / 2, 0, 0] }));
    group.add(mesh(new THREE.BoxGeometry(1.34, 0.16, 0.76), materials.fabricLight, { position: [0, -0.82, 0], rotation: [0, 0, -0.04] }));
    return group;
  },
  shoes() {
    const group = makeBase('#292524');
    group.add(mesh(new THREE.BoxGeometry(1.64, 0.26, 0.72), materials.black, { position: [0, -0.74, 0] }));
    group.add(mesh(new THREE.BoxGeometry(1.08, 0.58, 0.62), materials.stone, { position: [-0.22, -0.36, 0] }));
    group.add(mesh(new THREE.SphereGeometry(0.39, 32, 22), materials.fabricLight, { position: [0.66, -0.32, 0], scale: [1.18, 0.78, 0.86] }));
    group.add(mesh(new THREE.BoxGeometry(0.42, 0.52, 0.58), materials.mintDark, { position: [-0.72, 0.03, 0], rotation: [0, 0, -0.08] }));
    group.add(mesh(new THREE.TorusGeometry(0.36, 0.018, 10, 38), materials.label, { position: [0.05, -0.1, 0.33], rotation: [Math.PI / 2, 0.2, 0] }));
    return group;
  },
  biological() {
    const group = makeBase('#3f2b1d');
    group.add(mesh(new THREE.SphereGeometry(0.52, 36, 26), materials.foodGreen, { position: [-0.36, -0.36, 0], scale: [1, 0.8, 0.9] }));
    group.add(mesh(new THREE.SphereGeometry(0.46, 36, 26), materials.foodOrange, { position: [0.38, -0.28, 0.04], scale: [1.08, 0.82, 0.86] }));
    group.add(mesh(new THREE.ConeGeometry(0.34, 0.9, 36), materials.foodGreen, { position: [0, 0.42, 0], rotation: [0.7, 0.12, -0.5] }));
    group.add(mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.44, 16), materials.cardboardEdge, { position: [0.12, 0.52, 0.05], rotation: [0.5, 0, -0.3] }));
    return group;
  },
  trash() {
    const group = makeBase('#111827');
    group.add(mesh(new THREE.SphereGeometry(0.84, 42, 28), materials.black, { position: [0, -0.24, 0], scale: [1.06, 1.0, 0.9] }));
    group.add(mesh(new THREE.ConeGeometry(0.32, 0.78, 36), materials.black, { position: [-0.32, 0.48, 0], rotation: [0, 0, -0.36] }));
    group.add(mesh(new THREE.ConeGeometry(0.32, 0.78, 36), materials.black, { position: [0.32, 0.48, 0], rotation: [0, 0, 0.36] }));
    group.add(mesh(new THREE.BoxGeometry(0.62, 0.16, 0.08), materials.metalDark, { position: [0, 0.08, 0.62] }));
    return group;
  },
};

const outputs = {
  Clothes: builders.clothes,
  Glass: builders.glass,
  Plastic: builders.plastic,
  Shoes: builders.shoes,
  Cardboard: builders.cardboard,
  Paper: builders.paper,
  Metal: builders.metal,
  Battery: builders.battery,
  Biological: builders.biological,
  Trash: builders.trash,
};

async function exportGlb(name, build) {
  const scene = new THREE.Scene();
  const model = build();
  model.rotation.set(0.08, -0.38, 0);
  scene.add(model);
  const glb = await exporter.parseAsync(scene, { binary: true });
  await writeFile(resolve(outputDir, `${name}.glb`), Buffer.from(glb));
}

await mkdir(outputDir, { recursive: true });

for (const [name, build] of Object.entries(outputs)) {
  await exportGlb(name, build);
  console.log(`generated ${name}.glb`);
}
