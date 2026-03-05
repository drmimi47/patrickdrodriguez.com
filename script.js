import * as THREE from 'three';
import { MapControls } from 'three/addons/controls/MapControls.js';

import { createRadixSort } from '@three.ez/batched-mesh-extensions';

const instancesCount = 10000;
let instancesPerMesh;

// PlaneGeometry(2,2) with default 1x1 segments: 4 vertices, 6 indices
const PLANE_VERTEX_COUNT = 4;
const PLANE_INDEX_COUNT = 6;

const gridN = Math.ceil( Math.sqrt( instancesCount ) );
const gridSize = 5.5;
const gridTotal = gridN * gridSize;
const offsetX = new Float32Array( instancesCount );
const offsetZ = new Float32Array( instancesCount );
const meshInstanceIds = [];

let camera, scene, renderer, controls;
const batchedMeshes = [];
const originalTextures = [];
const blurredTextures = [];
const batchedMaterials = [];

const position = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const MAX_TILE_SIZE = 4;
const MAX_TEXTURE_SIZE = 512; // cap texture resolution to reduce GPU memory; lower = faster but more pixelated
const meshScales = [];
const matrix = new THREE.Matrix4();
const euler = new THREE.Euler( - Math.PI / 2, 0, 0 );

const keys = {};
const CAMERA_FLOOR = 1.0;  // minimum camera height above the tile plane
const ACCELERATION = 0.01; // world units added to speed per frame (constant ramp)
let moveSpeed = 0;
const velocity = new THREE.Vector3();
const _forward = new THREE.Vector3();
const _right = new THREE.Vector3();
const _move = new THREE.Vector3();
const _scrollPan = new THREE.Vector3();

let wasMoving = false;
let introActive = true;
let introStartTime = null;
const INTRO_DURATION = 4000;
const introStartPos = new THREE.Vector3( 0, 200, 550 );
const introEndPos = new THREE.Vector3( 0, 10, 23 );
// above is the camera start and end position for the intro animation
init();

// probe images/img1.png, img2.jpg, etc. until none found; supports .png and .jpg
async function discoverImages() {

	const paths = [];
	for ( let i = 1; ; i ++ ) {

		let found = false;
		for ( const ext of [ 'png', 'jpg' ] ) {

			try {

				const res = await fetch( `images/img${ i }.${ ext }`, { method: 'HEAD' } );
				if ( res.ok ) { paths.push( `images/img${ i }.${ ext }` ); found = true; break; }

			} catch {}

		}
		if ( ! found ) break;

	}
	return paths;

}

async function init() {

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setPixelRatio( window.devicePixelRatio );
	renderer.setSize( window.innerWidth, window.innerHeight );
	document.body.appendChild( renderer.domElement );

	scene = new THREE.Scene();
	scene.background = new THREE.Color( 0x111111 );
	scene.fog = new THREE.Fog( 0x111111, 100, 900 );

	camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 5000 );
	camera.position.copy( introStartPos );

	controls = new MapControls( camera, renderer.domElement );
	controls.maxPolarAngle = Math.PI / 2;
	controls.enableRotate = false;
	controls.enableZoom = false;
	controls.enabled = false;

	window.addEventListener( 'wheel', e => {

		if ( introActive ) return;
		camera.getWorldDirection( _scrollPan );
		_scrollPan.y = 0;
		_scrollPan.normalize();
		const scrollSpeed = Math.max( 0.5, camera.position.y * 0.05 );
		_right.crossVectors( _scrollPan, camera.up ).normalize();
		_scrollPan.multiplyScalar( - ( e.deltaY / 30 ) * scrollSpeed );
		_scrollPan.addScaledVector( _right, ( e.deltaX / 30 ) * scrollSpeed );
		camera.position.add( _scrollPan );
		controls.target.add( _scrollPan );
		controls.update();

	}, { passive: true } );

	const imagePaths = await discoverImages();
	instancesPerMesh = Math.ceil( instancesCount / imagePaths.length );

	const geometry = new THREE.PlaneGeometry( 2, 2 );
	const loader = new THREE.TextureLoader();

	quaternion.setFromEuler( euler );

	// shuffle grid positions so images are randomly distributed instead of banded
	const posOrder = new Int32Array( instancesCount );
	for ( let i = 0; i < instancesCount; i ++ ) posOrder[ i ] = i;
	for ( let i = instancesCount - 1; i > 0; i -- ) {
		const j = Math.floor( Math.random() * ( i + 1 ) );
		const tmp = posOrder[ i ]; posOrder[ i ] = posOrder[ j ]; posOrder[ j ] = tmp;
	}

	for ( let i = 0; i < imagePaths.length; i ++ ) {

		let texture = null;

		try {

			texture = await loader.loadAsync( imagePaths[ i ] );
			texture.colorSpace = THREE.SRGBColorSpace;
			texture.anisotropy = renderer.capabilities.getMaxAnisotropy();

		} catch ( e ) {

			// fallback to a solid colour if the image is missing
			console.warn( `Could not load ${ imagePaths[ i ] }`, e );

		}

		// downscale texture to MAX_TEXTURE_SIZE to save GPU memory
		if ( texture ) {

			const img = texture.image;
			const scale = Math.min( 1, MAX_TEXTURE_SIZE / Math.max( img.width, img.height ) );
			if ( scale < 1 ) {

				const tCanvas = document.createElement( 'canvas' );
				tCanvas.width = Math.round( img.width * scale );
				tCanvas.height = Math.round( img.height * scale );
				tCanvas.getContext( '2d' ).drawImage( img, 0, 0, tCanvas.width, tCanvas.height );
				texture.dispose();
				texture = new THREE.CanvasTexture( tCanvas );
				texture.colorSpace = THREE.SRGBColorSpace;
				texture.anisotropy = renderer.capabilities.getMaxAnisotropy();

			}

		}

		// store original and build a low-res blurred variant for use while moving
		originalTextures.push( texture );
		if ( texture ) {
			const img = texture.image;
			const bCanvas = document.createElement( 'canvas' );
			bCanvas.width = Math.max( 8, img.width >> 2 );
			bCanvas.height = Math.max( 8, img.height >> 2 );
			bCanvas.getContext( '2d' ).drawImage( img, 0, 0, bCanvas.width, bCanvas.height );
			const bt = new THREE.CanvasTexture( bCanvas );
			bt.colorSpace = THREE.SRGBColorSpace;
			blurredTextures.push( bt );
		} else {
			blurredTextures.push( null );
		}

		// compute tile scale: fit within MAX_TILE_SIZE, preserve aspect ratio
		// PlaneGeometry(2,2) has base size 2, so halfMax gives MAX_TILE_SIZE world units
		const halfMax = MAX_TILE_SIZE / 2;
		const aspect = texture ? texture.image.width / texture.image.height : 1;
		const tileScale = new THREE.Vector3(
			aspect >= 1 ? halfMax : halfMax * aspect,
			aspect >= 1 ? halfMax / aspect : halfMax,
			1
		);
		meshScales.push( tileScale );

		// MeshBasicMaterial shows the texture at full brightness without any lighting
		const material = new THREE.MeshBasicMaterial( {
			map: texture,
			color: texture ? 0xffffff : new THREE.Color( 0x00cc00 ),
			side: THREE.DoubleSide
		} );

		batchedMaterials.push( material );

		const mesh = new THREE.BatchedMesh( instancesPerMesh, PLANE_VERTEX_COUNT, PLANE_INDEX_COUNT, material );

		// enable radix sort for better performance
		mesh.customSort = createRadixSort( mesh );

		const geometryId = mesh.addGeometry( geometry );

		// place this mesh's slice of the global grid
		const ids = [];
		for ( let j = 0; j < instancesPerMesh; j ++ ) {

			const globalIndex = i * instancesPerMesh + j;
			if ( globalIndex >= instancesCount ) break;

			const gridIndex = posOrder[ globalIndex ];
			const col = gridIndex % gridN;
			const row = Math.floor( gridIndex / gridN );
			offsetX[ globalIndex ] = col * gridSize - gridTotal / 2 + gridSize / 2;
			offsetZ[ globalIndex ] = row * gridSize - gridTotal / 2 + gridSize / 2;

			position.set( offsetX[ globalIndex ], 0, offsetZ[ globalIndex ] );
			const id = mesh.addInstance( geometryId );
			ids.push( id );
			mesh.setMatrixAt( id, matrix.compose( position, quaternion, tileScale ) );

		}

		meshInstanceIds.push( ids );

		mesh.frustumCulled = false;
		mesh.perObjectFrustumCulled = false;
		scene.add( mesh );
		batchedMeshes.push( mesh );

	}

	document.addEventListener( 'keydown', e => { keys[ e.code ] = true; } );
	document.addEventListener( 'keyup', e => { keys[ e.code ] = false; } );
	window.addEventListener( 'resize', onWindowResize );

	let clickCount = 0;
	let helpTimeout = null;
	const helpOverlay = document.getElementById( 'help-overlay' );
	window.addEventListener( 'click', () => {

		clickCount ++;
		if ( clickCount === 3 ) {

			helpOverlay.style.display = 'flex';
			clearTimeout( helpTimeout );
			helpTimeout = setTimeout( () => {

				helpOverlay.style.display = 'none';
				clickCount = 0;

			}, 3000 );

		}

	} );
	onWindowResize();

	renderer.setAnimationLoop( animate );

}

function onWindowResize() {

	camera.aspect = window.innerWidth / window.innerHeight;
	camera.updateProjectionMatrix();

	renderer.setSize( window.innerWidth, window.innerHeight );

}

function animate() {

	if ( introActive ) {

		if ( introStartTime === null ) introStartTime = performance.now();
		const elapsed = performance.now() - introStartTime;
		let t = Math.min( elapsed / INTRO_DURATION, 1 );
		// ease-out cubic: fast approach, slow landing
		const te = 1 - Math.pow( 1 - t, 3 );

		camera.position.lerpVectors( introStartPos, introEndPos, te );
		camera.lookAt( 0, 0, 0 );

		if ( t >= 1 ) {

			introActive = false;
			controls.target.set( 0, 0, 0 );
			controls.update();
			controls.enabled = true;

		}

	} else {

		const maxSpeed = Math.max( 0.5, camera.position.y * 0.05 );

		const anyKey = keys[ 'KeyW' ] || keys[ 'ArrowUp' ] ||
		               keys[ 'KeyS' ] || keys[ 'ArrowDown' ] ||
		               keys[ 'KeyD' ] || keys[ 'ArrowRight' ] ||
		               keys[ 'KeyA' ] || keys[ 'ArrowLeft' ];

		if ( anyKey ) {
			moveSpeed = Math.min( moveSpeed + ACCELERATION, maxSpeed );
		} else {
			moveSpeed = 0;
		}

		camera.getWorldDirection( _forward );
		_forward.y = 0;
		_forward.normalize();
		_right.crossVectors( _forward, camera.up ).normalize();

		_move.set( 0, 0, 0 );
		if ( keys[ 'KeyW' ] || keys[ 'ArrowUp' ] )    _move.addScaledVector( _forward,  moveSpeed );
		if ( keys[ 'KeyS' ] || keys[ 'ArrowDown' ] )  _move.addScaledVector( _forward, -moveSpeed );
		if ( keys[ 'KeyD' ] || keys[ 'ArrowRight' ] ) _move.addScaledVector( _right,    moveSpeed );
		if ( keys[ 'KeyA' ] || keys[ 'ArrowLeft' ] )  _move.addScaledVector( _right,   -moveSpeed );

		velocity.lerp( _move, 0.1 );

		if ( velocity.lengthSq() > 0.0001 ) {

			camera.position.add( velocity );
			controls.target.add( velocity );
			controls.update();

		}

	}

	// keep camera above the tile plane
	if ( camera.position.y < CAMERA_FLOOR ) {
		camera.position.y = CAMERA_FLOOR;
		controls.target.y = Math.min( controls.target.y, camera.position.y - 0.01 );
		controls.update();
	}

	// wrap instances around camera for infinite tiling
	// during intro the camera is far back looking at origin, so tile around origin instead
	const camX = introActive ? 0 : camera.position.x;
	const camZ = introActive ? 0 : camera.position.z;

	for ( let meshIdx = 0; meshIdx < batchedMeshes.length; meshIdx ++ ) {

		const mesh = batchedMeshes[ meshIdx ];
		const ids = meshInstanceIds[ meshIdx ];

		for ( let j = 0; j < ids.length; j ++ ) {

			const globalIndex = meshIdx * instancesPerMesh + j;
			const ox = offsetX[ globalIndex ];
			const oz = offsetZ[ globalIndex ];
			const wx = ox + Math.round( ( camX - ox ) / gridTotal ) * gridTotal;
			const wz = oz + Math.round( ( camZ - oz ) / gridTotal ) * gridTotal;

			position.set( wx, 0, wz );
			mesh.setMatrixAt( ids[ j ], matrix.compose( position, quaternion, meshScales[ meshIdx ] ) );

		}

	}

	// swap to low-res texture per tile while moving, restore when stopped
	const isMoving = !introActive && velocity.lengthSq() > 0.01;
	if ( isMoving !== wasMoving ) {
		wasMoving = isMoving;
		for ( let i = 0; i < batchedMaterials.length; i ++ ) {
			if ( originalTextures[ i ] ) {
				batchedMaterials[ i ].map = isMoving ? blurredTextures[ i ] : originalTextures[ i ];
				batchedMaterials[ i ].needsUpdate = true;
			}
		}
	}

	renderer.render( scene, camera );

}
