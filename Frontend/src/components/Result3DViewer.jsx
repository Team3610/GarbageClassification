import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { Center, Clone, ContactShadows, Environment, OrbitControls, useGLTF } from '@react-three/drei';

const modelLabels = {
  Clothes: '의류',
  Glass: '유리',
  Plastic: '플라스틱',
  Shoes: '신발',
  Cardboard: '골판지',
  Paper: '종이',
  Metal: '금속',
  Battery: '배터리',
  Biological: '유기성 폐기물',
  Trash: '일반 쓰레기',
};

const modelDescriptions = {
  Clothes: '의류와 직물류를 대표하는 3D 모델',
  Glass: '유리병과 유리류를 대표하는 3D 모델',
  Plastic: '페트병과 플라스틱류를 대표하는 3D 모델',
  Shoes: '신발류를 대표하는 3D 모델',
  Cardboard: '택배 상자와 골판지를 대표하는 3D 모델',
  Paper: '종이 묶음을 대표하는 3D 모델',
  Metal: '알루미늄 캔과 금속류를 대표하는 3D 모델',
  Battery: '건전지와 배터리류를 대표하는 3D 모델',
  Biological: '음식물과 유기성 폐기물을 대표하는 3D 모델',
  Trash: '일반 쓰레기 봉투를 대표하는 3D 모델',
};

const modelPaths = {
  Clothes: '/models/3d/Clothes.glb',
  Glass: '/models/3d/Glass.glb',
  Plastic: '/models/3d/Plastic.glb',
  Shoes: '/models/3d/Shoes.glb',
  Cardboard: '/models/3d/Cardboard.glb',
  Paper: '/models/3d/Paper.glb',
  Metal: '/models/3d/Metal.glb',
  Battery: '/models/3d/Battery.glb',
  Biological: '/models/3d/Biological.glb',
  Trash: '/models/3d/Trash.glb',
};

// 결과가 나온 뒤 모델 로딩 지연으로 화면이 비는 시간을 줄이기 위해 대표 GLB를 미리 요청한다.
Object.values(modelPaths).forEach((path) => useGLTF.preload(path));

/**
 * 분류 결과 클래스에 맞는 대표 3D 모델을 보여준다.
 *
 * @param {{ prediction: { isGarbage: boolean, label?: string } | null }} props
 * @returns {JSX.Element}
 */
export function Result3DViewer({ prediction }) {
  const resultLabel = prediction?.isGarbage ? prediction.label : null;
  const previewLabel = resultLabel ?? (prediction ? null : 'Plastic');
  const modelPath = previewLabel ? modelPaths[previewLabel] : null;
  const title = resultLabel
    ? modelLabels[resultLabel] ?? resultLabel
    : prediction
      ? '비대상'
      : '예시: 플라스틱';
  const description = resultLabel
    ? modelDescriptions[resultLabel] ?? '분류 결과를 대표하는 3D 모델'
    : prediction
      ? '쓰레기로 인식되지 않아 대표 3D 모델을 표시하지 않습니다.'
      : '업로드 전 확인용 예시 모델입니다. 분류가 끝나면 결과 클래스에 맞는 모델로 자동 교체됩니다.';

  return (
    <div className="relative aspect-[4/3] overflow-hidden rounded-[24px] border border-white/10 bg-[#101917]">
      <Canvas
        className="h-full w-full"
        camera={{ position: [3.2, 2.4, 4.2], fov: 42 }}
        dpr={[1, 1.8]}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={['#111817']} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[4, 6, 4]} intensity={2.6} />
        <pointLight position={[-3, 2, -2]} intensity={1.3} color="#9ed7bd" />
        <Environment preset="warehouse" />
        <Suspense fallback={<WaitingModel />}>
          <Center>
            {modelPath ? <ClassModel path={modelPath} /> : <WaitingModel />}
          </Center>
        </Suspense>
        <ContactShadows position={[0, -1.28, 0]} opacity={0.45} blur={2.5} scale={8} />
        <OrbitControls
          enablePan={false}
          enableDamping
          dampingFactor={0.08}
          minDistance={2.6}
          maxDistance={6}
          maxPolarAngle={Math.PI / 2.05}
        />
      </Canvas>
      <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-[#101917]/95 via-[#101917]/55 to-transparent p-4 pt-16">
        <p className="text-xs font-semibold text-white/70">마우스로 회전 / 휠로 확대</p>
        <p className="text-lg font-black">{title}</p>
        <p className="mt-1 text-sm leading-5 text-white/70">{description}</p>
      </div>
    </div>
  );
}

function ClassModel({ path }) {
  const { scene } = useGLTF(path);

  return (
    <group scale={1.12} rotation={[0.02, 0, 0]}>
      <Clone object={scene} />
    </group>
  );
}

function WaitingModel() {
  return (
    <group rotation={[0.12, -0.55, 0]}>
      <mesh position={[0, -0.72, 0]}>
        <boxGeometry args={[1.9, 0.2, 1.4]} />
        <meshStandardMaterial color="#2f4f45" roughness={0.7} />
      </mesh>
      <mesh position={[-0.55, -0.08, 0]} rotation={[0, 0, -0.1]}>
        <boxGeometry args={[0.68, 1.08, 0.8]} />
        <meshStandardMaterial color="#7bd6ac" roughness={0.45} />
      </mesh>
      <mesh position={[0.55, -0.08, 0]} rotation={[0, 0, 0.1]}>
        <boxGeometry args={[0.68, 1.08, 0.8]} />
        <meshStandardMaterial color="#b7e6d0" roughness={0.45} />
      </mesh>
      <mesh position={[0, 0.72, 0]}>
        <torusGeometry args={[0.56, 0.08, 16, 48]} />
        <meshStandardMaterial color="#ecfdf3" roughness={0.36} />
      </mesh>
    </group>
  );
}
