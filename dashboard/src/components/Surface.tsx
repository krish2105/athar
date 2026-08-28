import { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

/** The budget surface, in three dimensions because it genuinely has three.
 *
 * Two channels vary across their permitted spend range; the remaining budget is
 * split among the other three in proportion to the observed plan. Height is the
 * revenue the whole allocation earns under the true response curves.
 *
 * This is the one place on the page where a third dimension shows something a
 * flat chart cannot: the optima chosen by different estimators are points on a
 * shared curved landscape, and how far each sits down the slope from the ridge
 * *is* the cost of believing it. A heatmap would show the height; it would not
 * show that the surface is nearly flat along one direction and steep along
 * another, which is why two estimators can disagree wildly about the split and
 * barely differ in revenue — or the reverse.
 */

type SurfaceData = {
  channels: [string, string];
  axis_spend: number[];
  revenue: (number | null)[][];
};

type Marker = { key: string; label: string; x: number; y: number; z: number; color: string };

function Mesh({
  data,
  markers,
  dark,
}: {
  data: SurfaceData;
  markers: Marker[];
  dark: boolean;
}) {
  const { geometry, scaleZ, minZ } = useMemo(() => {
    const n = data.axis_spend.length;
    const flat = data.revenue.flat().filter((v): v is number => v != null);
    const minZ = Math.min(...flat);
    const maxZ = Math.max(...flat);
    const scaleZ = 1 / (maxZ - minZ);

    const geometry = new THREE.PlaneGeometry(2, 2, n - 1, n - 1);
    const position = geometry.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(position.count * 3);
    const low = new THREE.Color(dark ? "#1d2a3d" : "#dbe6f5");
    const high = new THREE.Color(dark ? "#5aa0f0" : "#1c5fae");

    for (let i = 0; i < position.count; i++) {
      const row = Math.floor(i / n);
      const column = i % n;
      const value = data.revenue[row]?.[column];
      const height = value == null ? 0 : (value - minZ) * scaleZ;
      position.setZ(i, height * 0.85);
      const shade = low.clone().lerp(high, value == null ? 0 : height);
      colors[i * 3] = shade.r;
      colors[i * 3 + 1] = shade.g;
      colors[i * 3 + 2] = shade.b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    return { geometry, scaleZ, minZ };
  }, [data, dark]);

  const group = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (group.current) group.current.rotation.z += delta * 0.045;
  });

  const span = data.axis_spend[data.axis_spend.length - 1] - data.axis_spend[0];
  const place = (marker: Marker): [number, number, number] => [
    ((marker.x - data.axis_spend[0]) / span) * 2 - 1,
    ((marker.y - data.axis_spend[0]) / span) * 2 - 1,
    (marker.z - minZ) * scaleZ * 0.85 + 0.045,
  ];

  return (
    <group ref={group} rotation={[-Math.PI / 2.42, 0, 0.5]}>
      <mesh geometry={geometry}>
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={0.72}
          metalness={0.06}
          flatShading={false}
        />
      </mesh>
      <lineSegments>
        <wireframeGeometry args={[geometry]} />
        <lineBasicMaterial
          color={dark ? "#3a4657" : "#ffffff"}
          transparent
          opacity={dark ? 0.16 : 0.3}
        />
      </lineSegments>
      {markers.map((marker) => {
        const position = place(marker);
        return (
          <group key={marker.key} position={position}>
            <mesh>
              <sphereGeometry args={[0.038, 20, 20]} />
              <meshStandardMaterial color={marker.color} roughness={0.35} />
            </mesh>
            <mesh position={[0, 0, -position[2] / 2]}>
              <cylinderGeometry args={[0.004, 0.004, position[2], 8]} />
              <meshBasicMaterial color={marker.color} transparent opacity={0.5} />
            </mesh>
            <Html center distanceFactor={4.2} style={{ pointerEvents: "none" }}>
              <div
                style={{
                  transform: "translateY(-22px)",
                  whiteSpace: "nowrap",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 9,
                  letterSpacing: ".06em",
                  textTransform: "uppercase",
                  color: marker.color,
                  background: dark ? "rgba(22,23,27,.82)" : "rgba(250,248,244,.86)",
                  padding: "2px 5px",
                  borderRadius: 2,
                }}
              >
                {marker.label}
              </div>
            </Html>
          </group>
        );
      })}
    </group>
  );
}

function Rig() {
  const { camera } = useThree();
  camera.position.set(0, -3.0, 2.15);
  camera.lookAt(0, 0, 0.25);
  return null;
}

export function BudgetSurface({
  data,
  markers,
  dark,
}: {
  data: SurfaceData;
  markers: Marker[];
  dark: boolean;
}) {
  return (
    <div className="surface-wrap">
      <Canvas dpr={[1, 2]} camera={{ fov: 38 }} gl={{ antialias: true }}>
        <Rig />
        <ambientLight intensity={dark ? 0.55 : 0.78} />
        <directionalLight position={[3, -4, 6]} intensity={dark ? 1.5 : 1.15} />
        <directionalLight position={[-4, 2, 3]} intensity={0.32} />
        <Mesh data={data} markers={markers} dark={dark} />
        <OrbitControls
          enablePan={false}
          minDistance={2.2}
          maxDistance={6}
          maxPolarAngle={Math.PI / 2.05}
        />
      </Canvas>
      <div className="surface-hint">drag to rotate · scroll to zoom</div>
    </div>
  );
}
