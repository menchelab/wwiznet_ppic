import{ge as h$,gf as d$}from"./index-D1nhnm6i.js";function p$(n,t){for(var e=0;e<t.length;e++){const s=t[e];if(typeof s!="string"&&!Array.isArray(s)){for(const o in s)if(o!=="default"&&!(o in n)){const r=Object.getOwnPropertyDescriptor(s,o);r&&Object.defineProperty(n,o,r.get?r:{enumerable:!0,get:()=>s[o]})}}}return Object.freeze(Object.defineProperty(n,Symbol.toStringTag,{value:"Module"}))}const f$=1e-7,m$=1e-4;class Gm{constructor(t,e){this.backend=t,this.dataMover=e,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,e){this.dataIdsCount++,this.data.set(t,e)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Yu{refCount(t){return He("refCount")}incRef(t){return He("incRef")}timerAvailable(){return!0}time(t){return He("time")}read(t){return He("read")}readSync(t){return He("readSync")}readToGPU(t,e){return He("readToGPU")}numDataIds(){return He("numDataIds")}disposeData(t,e){return He("disposeData")}write(t,e,s){return He("write")}move(t,e,s,o,r){return He("move")}createTensorFromGPUData(t,e,s){return He("createTensorFromGPUData")}memory(){return He("memory")}floatPrecision(){return He("floatPrecision")}epsilon(){return this.floatPrecision()===32?f$:m$}dispose(){return He("dispose")}}function He(n){throw new Error(`'${n}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}function Zu(n){let t=n.length,e=0;for(;t>0;)e=Math.random()*t|0,t--,Jn(n,t,e)}function g$(n,t){if(n.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${n.length}Second array length was ${t.length}`);let e=n.length,s=0;for(;e>0;)s=Math.random()*e|0,e--,Jn(n,e,s),Jn(t,e,s)}function Ns(n,t,e){return Math.max(n,Math.min(t,e))}function fl(n){return n%2===0?n:n+1}function Jn(n,t,e){const s=n[t];n[t]=n[e],n[e]=s}function Hm(n){let t=0;for(let e=0;e<n.length;e++)t+=n[e];return t}function x$(n,t){const e=Math.random();return t*e+(1-e)*n}function b$(n,t){let e=0;for(let s=0;s<n.length;s++){const o=Number(n[s])-Number(t[s]);e+=o*o}return e}function I(n,t){if(!n)throw new Error(typeof t=="string"?t:t())}function Oe(n,t,e=""){I(Tt(n,t),()=>e+` Shapes ${n} and ${t} must match`)}function Ju(n){I(n!=null,()=>"The input to the tensor constructor must be a non-null value.")}function U(n){if(n.length===0)return 1;let t=n[0];for(let e=1;e<n.length;e++)t*=n[e];return t}function y$(n){return n.length===0}function w$(n,t){if(n===t)return!0;if(n==null||t==null||n.length!==t.length)return!1;for(let e=0;e<n.length;e++)if(n[e]!==null&&t[e]!==null&&n[e]!==t[e])return!1;return!0}function Tt(n,t){if(n===t)return!0;if(n==null||t==null||n.length!==t.length)return!1;for(let e=0;e<n.length;e++)if(n[e]!==t[e])return!1;return!0}function oo(n){return n%1===0}function $$(n){if(Math.tanh!=null)return Math.tanh(n);if(n===1/0)return 1;if(n===-1/0)return-1;{const t=Math.exp(2*n);return(t-1)/(t+1)}}function za(n){const t=Math.ceil(Math.sqrt(n));return[t,Math.ceil(n/t)]}function C$(n){const t=new Uint32Array(n);for(let e=0;e<n;++e)t[e]=e;return Zu(t),t}function to(n,t){return t<=n.length?n:n+" ".repeat(t-n.length)}function pu(n,t=o=>0,e,s){return new Promise((o,r)=>{let i=0;const a=()=>{if(n()){o();return}i++;const l=t(i);if(e!=null&&i>=e){r();return}s!=null?s(a,l):setTimeout(a,l)};a()})}function Qu(n,t){let e=1,s=-1;for(let r=0;r<n.length;++r)if(n[r]>=0)e*=n[r];else if(n[r]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${r}`);s=r}else if(n[r]<0)throw Error(`Shapes can not be < 0. Found ${n[r]} at dim ${r}`);if(s===-1){if(t>0&&t!==e)throw Error(`Size(${t}) must match the product of shape ${n}`);return n}if(e===0)throw Error(`Cannot infer the missing size in [${n}] when there are 0 elements`);if(t%e!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${e}`);const o=n.slice();return o[s]=t/e,o}function $t(n,t){const e=t.length;return n=n==null?t.map((s,o)=>o):[].concat(n),I(n.every(s=>s>=-e&&s<e),()=>`All values in axis param must be in range [-${e}, ${e}) but got axis ${n}`),I(n.every(s=>oo(s)),()=>`All values in axis param must be integers but got axis ${n}`),n.map(s=>s<0?e+s:s)}function is(n,t){const e=[],s=[],o=t!=null&&Array.isArray(t)&&t.length===0,r=t==null||o?null:$t(t,n).sort();let i=0;for(let a=0;a<n.length;++a){if(r!=null){if(r[i]===a&&n[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${n[a]}' is not 1`);(r[i]==null||r[i]>a)&&n[a]===1&&(e.push(n[a]),s.push(a)),r[i]<=a&&i++}n[a]!==1&&(e.push(n[a]),s.push(a))}return{newShape:e,keptDims:s}}function we(n,t){return Jt(n,t)}function Jt(n,t){let e=null;if(n==null||n==="float32")e=new Float32Array(t);else if(n==="int32")e=new Int32Array(t);else if(n==="bool")e=new Uint8Array(t);else if(n==="string")e=new Array(t);else throw new Error(`Unknown data type ${n}`);return e}function qm(n,t){for(let e=0;e<n.length;e++){const s=n[e];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function jm(n){return n==="bool"||n==="complex64"||n==="float32"||n==="int32"||n==="string"}function th(n,t){return!(t==="complex64"||t==="float32"&&n!=="complex64"||t==="int32"&&n!=="float32"&&n!=="complex64"||t==="bool"&&n==="bool")}function Ar(n){if(n==="float32"||n==="int32")return 4;if(n==="complex64")return 8;if(n==="bool")return 1;throw new Error(`Unknown dtype ${n}`)}function Km(n){if(n==null)return 0;let t=0;return n.forEach(e=>t+=e.length),t}function Qo(n){return typeof n=="string"||n instanceof String}function Xm(n){return typeof n=="boolean"}function Ba(n){return typeof n=="number"}function vo(n){return Array.isArray(n)?vo(n[0]):n instanceof Float32Array?"float32":n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray?"int32":Ba(n)?"float32":Qo(n)?"string":Xm(n)?"bool":"float32"}function Ts(n){return!!(n&&n.constructor&&n.call&&n.apply)}function Va(n,t){for(let e=t;e<n;++e)if(n%e===0)return e;return n}function ut(n){const t=n.length;if(t<2)return[];const e=new Array(t-1);e[t-2]=n[t-1];for(let s=t-3;s>=0;--s)e[s]=e[s+1]*n[s+1];return e}function Ym(n,t,e,s=!1){const o=new Array;if(t.length===1){const r=t[0]*(s?2:1);for(let i=0;i<r;i++)o[i]=e[n+i]}else{const r=t[0],i=t.slice(1),a=i.reduce((l,c)=>l*c)*(s?2:1);for(let l=0;l<r;l++)o[l]=Ym(n+l*a,i,e,s)}return o}function pn(n,t,e=!1){if(n.length===0)return t[0];const s=n.reduce((o,r)=>o*r)*(e?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${n}] does not match the input size ${t.length}${e?" for a complex tensor":""}.`);return Ym(0,n,t,e)}function Zm(n,t){if(Array.isArray(n))return n;if(t==="float32")return n instanceof Float32Array?n:new Float32Array(n);if(t==="int32")return n instanceof Int32Array?n:new Int32Array(n);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(n));throw new Error(`Unknown dtype ${t}`)}function ml(n,t){const e=$e(n,t);for(let s=0;s<e.length;s++)e[s]=1;return e}function $e(n,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(n);if(t==="int32")return new Int32Array(n);if(t==="bool")return new Uint8Array(n);throw new Error(`Unknown data type ${t}`)}function eh(n,t){const e=n.reduce((s,o)=>s*o,1);if(t==null||t==="float32")return pn(n,new Float32Array(e));if(t==="int32")return pn(n,new Int32Array(e));if(t==="bool")return pn(n,new Uint8Array(e));throw new Error(`Unknown data type ${t}`)}function Fn(n){n.forEach(t=>{I(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${n}].`)})}function Sn(n,t,e){if(t===0)return 0;if(t===1)return n[0];let s=n[n.length-1];for(let o=0;o<n.length-1;++o)s+=e[o]*n[o];return s}function ko(n,t,e){if(t===0)return[];if(t===1)return[n];const s=new Array(t);for(let o=0;o<s.length-1;++o)s[o]=Math.floor(n/e[o]),n-=s[o]*e[o];return s[s.length-1]=n,s}function Kr(n){return n&&n.then&&typeof n.then=="function"}const rf="tfjsflags";class I${constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=v$,this.populateURLFlags()}setPlatform(t,e){this.platform!=null&&(B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=e}registerFlag(t,e,s){if(this.flagRegistry[t]={evaluationFn:e,setHook:s},this.urlFlags[t]!=null){const o=this.urlFlags[t];B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${o}.`),this.set(t,o)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const e=this.evaluateFlag(t);if(Kr(e))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=e,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,e){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=e,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(e)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);rf in t&&t[rf].split(",").forEach(s=>{const[o,r]=s.split(":");this.urlFlags[o]=S$(o,r)})}}function v$(n){const t={};return n.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(e,...s)=>(k$(t,s[0],s[1]),s.join("="))),t}function k$(n,t,e){n[decodeURIComponent(t)]=decodeURIComponent(e||"")}function S$(n,t){const e=t.toLowerCase();return e==="true"||e==="false"?e==="true":`${+e}`===e?+e:t}function B(){return Jm}let Jm=null;function N$(n){Jm=n}let Jc;function Qm(){if(Jc==null){let n;if(typeof window<"u")n=window;else if(typeof global<"u")n=global;else if(typeof process<"u")n=process;else if(typeof self<"u")n=self;else throw new Error("Could not find a global object");Jc=n}return Jc}function T$(){const n=Qm();return n._tfGlobals==null&&(n._tfGlobals=new Map),n._tfGlobals}function nh(n,t){const e=T$();if(e.has(n))return e.get(n);{const s=t();return e.set(n,s),e.get(n)}}const gl="Abs",Xr="Acos",Yr="Acosh",tr="Add",sh="AddN",oh="All",rh="Any",xl="ArgMax",bl="ArgMin",Zr="Asin",Jr="Asinh",Qr="Atan",ti="Atanh",ei="Atan2",yl="AvgPool",ih="AvgPoolGrad",wl="AvgPool3D",ah="AvgPool3DGrad",$l="BatchMatMul",Cl="BatchToSpaceND",lh="Bincount",ch="BitwiseAnd",E$="BroadcastTo",tg="BroadcastArgs",ni="Cast",si="Ceil",oi="ClipByValue",uh="Complex",Il="ComplexAbs",vl="Concat",kl="Conv2D",hh="Conv2DBackpropFilter",Sl="Conv2DBackpropInput",Nl="Conv3D",dh="Conv3DBackpropFilterV2",ph="Conv3DBackpropInputV2",ri="Cos",ii="Cosh",fh="Cumprod",Tl="Cumsum",mh="CropAndResize",gh="DenseBincount",xh="DepthToSpace",El="DepthwiseConv2dNative",bh="DepthwiseConv2dNativeBackpropFilter",yh="DepthwiseConv2dNativeBackpropInput",eg="Diag",Rl="Dilation2D",fu="Dilation2DBackpropInput",mu="Dilation2DBackpropFilter",R$="Draw",ai="RealDiv",wh="Einsum",li="Elu",$h="EluGrad",ci="Erf",Al="Equal",ui="Exp",Dl="ExpandDims",hi="Expm1",Ch="FFT",Ih="Fill",vh="FlipLeftRight",di="Floor",pi="FloorDiv",Fl="FusedBatchNorm",_l="GatherV2",kh="GatherNd",Ol="Greater",fi="GreaterEqual",mi="Identity",Sh="IFFT",Nh="Imag",gi="IsFinite",xi="IsInf",bi="IsNan",Ll="LeakyRelu",Ml="Less",Pl="LessEqual",ng="LinSpace",yi="Log",wi="Log1p",zl="LogicalAnd",Bl="LogicalNot",Vl="LogicalOr",pK="LogicalXor",A$="LogSoftmax",fK="LowerBound",Wl="LRN",Th="LRNGrad",mK="MatrixBandPart",Ul="Max",$i="Maximum",Gl="MaxPool",Eh="MaxPoolGrad",Hl="MaxPool3D",Rh="MaxPool3DGrad",sg="MaxPoolWithArgmax",ql="Mean",jl="Min",Ci="Minimum",Kl="MirrorPad",Ii="Mod",og="Multinomial",vi="Multiply",Xl="Neg",Yl="NotEqual",Ah="NonMaxSuppressionV3",Dh="NonMaxSuppressionV4",Fh="NonMaxSuppressionV5",Zl="OnesLike",Jl="OneHot",Ql="Pack",tc="PadV2",gK="Pool",ki="Pow",ec="Prelu",nc="Prod",rg="RaggedGather",ig="RaggedRange",ag="RaggedTensorToTensor",_h="Range",Oh="Real",Si="Reciprocal",Ni="Relu",sc="Reshape",oc="ResizeNearestNeighbor",Lh="ResizeNearestNeighborGrad",rc="ResizeBilinear",Mh="ResizeBilinearGrad",Ti="Relu6",ic="Reverse",Ei="Round",Ri="Rsqrt",lg="ScatterNd",cg="TensorScatterUpdate",ug="SearchSorted",ac="Select",Ai="Selu",lc="Slice",Di="Sin",Fi="Sinh",_i="Sign",Oi="Sigmoid",Li="Softplus",Mi="Sqrt",cc="Sum",uc="SpaceToBatchND",hc="SplitV",dc="Softmax",Ph="SparseFillEmptyRows",zh="SparseReshape",Bh="SparseSegmentMean",Vh="SparseSegmentSum",hg="SparseToDense",Pi="SquaredDifference",Wh="Square",pc="StaticRegexReplace",Uh="StridedSlice",Gh="StringNGrams",Hh="StringSplit",qh="StringToHashBucketFast",zi="Sub",Bi="Tan",Vi="Tanh",Wi="Tile",jh="TopK",Kh="Transform",zo="Transpose",Xh="Unique",fc="Unpack",mc="UnsortedSegmentSum",xK="UpperBound",gc="ZerosLike",Ui="Step",D$="FromPixels",Yh="RotateWithOffset",Wa="_FusedMatMul",Ua="FusedConv2D",dg="FusedDepthwiseConv2D";function Qe(...n){B().getBool("IS_TEST")||B().getBool("PROD")||console.warn(...n)}function F$(...n){B().getBool("IS_TEST")||B().getBool("PROD")||console.log(...n)}const Ho=nh("kernelRegistry",()=>new Map),Dr=nh("gradRegistry",()=>new Map);function af(n,t){const e=Jh(n,t);return Ho.get(e)}function lf(n){return Dr.get(n)}function gu(n){const t=Ho.entries(),e=[];for(;;){const{done:s,value:o}=t.next();if(s)break;const[r,i]=o,[a]=r.split("_");a===n&&e.push(i)}return e}function Zh(n){const{kernelName:t,backendName:e}=n,s=Jh(t,e);Ho.has(s)&&Qe(`The kernel '${t}' for backend '${e}' is already registered`),Ho.set(s,n)}function _$(n){const{kernelName:t}=n;Dr.has(t)&&B().getBool("DEBUG")&&Qe(`Overriding the gradient for '${t}'`),Dr.set(t,n)}function bK(n,t){const e=Jh(n,t);if(!Ho.has(e))throw new Error(`The kernel '${n}' for backend '${t}' is not registered`);Ho.delete(e)}function yK(n){if(!Dr.has(n))throw new Error(`The gradient '${n}' for backend is not registered`);Dr.delete(n)}function wK(n,t){gu(n).forEach(s=>{const o=Object.assign({},s,{backendName:t});Zh(o)})}function Jh(n,t){return`${t}_${n}`}function pg(n){return n instanceof Float32Array||n instanceof Int32Array||n instanceof Uint8Array||n instanceof Uint8ClampedArray}var Qc,cf;function O$(){if(cf)return Qc;cf=1,Qc=t;var n=null;try{n=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function t(S,C,D){this.low=S|0,this.high=C|0,this.unsigned=!!D}t.prototype.__isLong__,Object.defineProperty(t.prototype,"__isLong__",{value:!0});function e(S){return(S&&S.__isLong__)===!0}t.isLong=e;var s={},o={};function r(S,C){var D,O,P;return C?(S>>>=0,(P=0<=S&&S<256)&&(O=o[S],O)?O:(D=a(S,(S|0)<0?-1:0,!0),P&&(o[S]=D),D)):(S|=0,(P=-128<=S&&S<128)&&(O=s[S],O)?O:(D=a(S,S<0?-1:0,!1),P&&(s[S]=D),D))}t.fromInt=r;function i(S,C){if(isNaN(S))return C?b:x;if(C){if(S<0)return b;if(S>=f)return N}else{if(S<=-m)return T;if(S+1>=m)return v}return S<0?i(-S,C).neg():a(S%p|0,S/p|0,C)}t.fromNumber=i;function a(S,C,D){return new t(S,C,D)}t.fromBits=a;var l=Math.pow;function c(S,C,D){if(S.length===0)throw Error("empty string");if(S==="NaN"||S==="Infinity"||S==="+Infinity"||S==="-Infinity")return x;if(typeof C=="number"?(D=C,C=!1):C=!!C,D=D||10,D<2||36<D)throw RangeError("radix");var O;if((O=S.indexOf("-"))>0)throw Error("interior hyphen");if(O===0)return c(S.substring(1),C,D).neg();for(var P=i(l(D,8)),M=x,V=0;V<S.length;V+=8){var H=Math.min(8,S.length-V),W=parseInt(S.substring(V,V+H),D);if(H<8){var q=i(l(D,H));M=M.mul(q).add(i(W))}else M=M.mul(P),M=M.add(i(W))}return M.unsigned=C,M}t.fromString=c;function u(S,C){return typeof S=="number"?i(S,C):typeof S=="string"?c(S,C):a(S.low,S.high,typeof C=="boolean"?C:S.unsigned)}t.fromValue=u;var h=65536,d=1<<24,p=h*h,f=p*p,m=f/2,g=r(d),x=r(0);t.ZERO=x;var b=r(0,!0);t.UZERO=b;var w=r(1);t.ONE=w;var y=r(1,!0);t.UONE=y;var $=r(-1);t.NEG_ONE=$;var v=a(-1,2147483647,!1);t.MAX_VALUE=v;var N=a(-1,-1,!0);t.MAX_UNSIGNED_VALUE=N;var T=a(0,-2147483648,!1);t.MIN_VALUE=T;var k=t.prototype;return k.toInt=function(){return this.unsigned?this.low>>>0:this.low},k.toNumber=function(){return this.unsigned?(this.high>>>0)*p+(this.low>>>0):this.high*p+(this.low>>>0)},k.toString=function(C){if(C=C||10,C<2||36<C)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(T)){var D=i(C),O=this.div(D),P=O.mul(D).sub(this);return O.toString(C)+P.toInt().toString(C)}else return"-"+this.neg().toString(C);for(var M=i(l(C,6),this.unsigned),V=this,H="";;){var W=V.div(M),q=V.sub(W.mul(M)).toInt()>>>0,j=q.toString(C);if(V=W,V.isZero())return j+H;for(;j.length<6;)j="0"+j;H=""+j+H}},k.getHighBits=function(){return this.high},k.getHighBitsUnsigned=function(){return this.high>>>0},k.getLowBits=function(){return this.low},k.getLowBitsUnsigned=function(){return this.low>>>0},k.getNumBitsAbs=function(){if(this.isNegative())return this.eq(T)?64:this.neg().getNumBitsAbs();for(var C=this.high!=0?this.high:this.low,D=31;D>0&&(C&1<<D)==0;D--);return this.high!=0?D+33:D+1},k.isZero=function(){return this.high===0&&this.low===0},k.eqz=k.isZero,k.isNegative=function(){return!this.unsigned&&this.high<0},k.isPositive=function(){return this.unsigned||this.high>=0},k.isOdd=function(){return(this.low&1)===1},k.isEven=function(){return(this.low&1)===0},k.equals=function(C){return e(C)||(C=u(C)),this.unsigned!==C.unsigned&&this.high>>>31===1&&C.high>>>31===1?!1:this.high===C.high&&this.low===C.low},k.eq=k.equals,k.notEquals=function(C){return!this.eq(C)},k.neq=k.notEquals,k.ne=k.notEquals,k.lessThan=function(C){return this.comp(C)<0},k.lt=k.lessThan,k.lessThanOrEqual=function(C){return this.comp(C)<=0},k.lte=k.lessThanOrEqual,k.le=k.lessThanOrEqual,k.greaterThan=function(C){return this.comp(C)>0},k.gt=k.greaterThan,k.greaterThanOrEqual=function(C){return this.comp(C)>=0},k.gte=k.greaterThanOrEqual,k.ge=k.greaterThanOrEqual,k.compare=function(C){if(e(C)||(C=u(C)),this.eq(C))return 0;var D=this.isNegative(),O=C.isNegative();return D&&!O?-1:!D&&O?1:this.unsigned?C.high>>>0>this.high>>>0||C.high===this.high&&C.low>>>0>this.low>>>0?-1:1:this.sub(C).isNegative()?-1:1},k.comp=k.compare,k.negate=function(){return!this.unsigned&&this.eq(T)?T:this.not().add(w)},k.neg=k.negate,k.add=function(C){e(C)||(C=u(C));var D=this.high>>>16,O=this.high&65535,P=this.low>>>16,M=this.low&65535,V=C.high>>>16,H=C.high&65535,W=C.low>>>16,q=C.low&65535,j=0,K=0,Y=0,Z=0;return Z+=M+q,Y+=Z>>>16,Z&=65535,Y+=P+W,K+=Y>>>16,Y&=65535,K+=O+H,j+=K>>>16,K&=65535,j+=D+V,j&=65535,a(Y<<16|Z,j<<16|K,this.unsigned)},k.subtract=function(C){return e(C)||(C=u(C)),this.add(C.neg())},k.sub=k.subtract,k.multiply=function(C){if(this.isZero())return x;if(e(C)||(C=u(C)),n){var D=n.mul(this.low,this.high,C.low,C.high);return a(D,n.get_high(),this.unsigned)}if(C.isZero())return x;if(this.eq(T))return C.isOdd()?T:x;if(C.eq(T))return this.isOdd()?T:x;if(this.isNegative())return C.isNegative()?this.neg().mul(C.neg()):this.neg().mul(C).neg();if(C.isNegative())return this.mul(C.neg()).neg();if(this.lt(g)&&C.lt(g))return i(this.toNumber()*C.toNumber(),this.unsigned);var O=this.high>>>16,P=this.high&65535,M=this.low>>>16,V=this.low&65535,H=C.high>>>16,W=C.high&65535,q=C.low>>>16,j=C.low&65535,K=0,Y=0,Z=0,tt=0;return tt+=V*j,Z+=tt>>>16,tt&=65535,Z+=M*j,Y+=Z>>>16,Z&=65535,Z+=V*q,Y+=Z>>>16,Z&=65535,Y+=P*j,K+=Y>>>16,Y&=65535,Y+=M*q,K+=Y>>>16,Y&=65535,Y+=V*W,K+=Y>>>16,Y&=65535,K+=O*j+P*q+M*W+V*H,K&=65535,a(Z<<16|tt,K<<16|Y,this.unsigned)},k.mul=k.multiply,k.divide=function(C){if(e(C)||(C=u(C)),C.isZero())throw Error("division by zero");if(n){if(!this.unsigned&&this.high===-2147483648&&C.low===-1&&C.high===-1)return this;var D=(this.unsigned?n.div_u:n.div_s)(this.low,this.high,C.low,C.high);return a(D,n.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:x;var O,P,M;if(this.unsigned){if(C.unsigned||(C=C.toUnsigned()),C.gt(this))return b;if(C.gt(this.shru(1)))return y;M=b}else{if(this.eq(T)){if(C.eq(w)||C.eq($))return T;if(C.eq(T))return w;var V=this.shr(1);return O=V.div(C).shl(1),O.eq(x)?C.isNegative()?w:$:(P=this.sub(C.mul(O)),M=O.add(P.div(C)),M)}else if(C.eq(T))return this.unsigned?b:x;if(this.isNegative())return C.isNegative()?this.neg().div(C.neg()):this.neg().div(C).neg();if(C.isNegative())return this.div(C.neg()).neg();M=x}for(P=this;P.gte(C);){O=Math.max(1,Math.floor(P.toNumber()/C.toNumber()));for(var H=Math.ceil(Math.log(O)/Math.LN2),W=H<=48?1:l(2,H-48),q=i(O),j=q.mul(C);j.isNegative()||j.gt(P);)O-=W,q=i(O,this.unsigned),j=q.mul(C);q.isZero()&&(q=w),M=M.add(q),P=P.sub(j)}return M},k.div=k.divide,k.modulo=function(C){if(e(C)||(C=u(C)),n){var D=(this.unsigned?n.rem_u:n.rem_s)(this.low,this.high,C.low,C.high);return a(D,n.get_high(),this.unsigned)}return this.sub(this.div(C).mul(C))},k.mod=k.modulo,k.rem=k.modulo,k.not=function(){return a(~this.low,~this.high,this.unsigned)},k.and=function(C){return e(C)||(C=u(C)),a(this.low&C.low,this.high&C.high,this.unsigned)},k.or=function(C){return e(C)||(C=u(C)),a(this.low|C.low,this.high|C.high,this.unsigned)},k.xor=function(C){return e(C)||(C=u(C)),a(this.low^C.low,this.high^C.high,this.unsigned)},k.shiftLeft=function(C){return e(C)&&(C=C.toInt()),(C&=63)===0?this:C<32?a(this.low<<C,this.high<<C|this.low>>>32-C,this.unsigned):a(0,this.low<<C-32,this.unsigned)},k.shl=k.shiftLeft,k.shiftRight=function(C){return e(C)&&(C=C.toInt()),(C&=63)===0?this:C<32?a(this.low>>>C|this.high<<32-C,this.high>>C,this.unsigned):a(this.high>>C-32,this.high>=0?0:-1,this.unsigned)},k.shr=k.shiftRight,k.shiftRightUnsigned=function(C){if(e(C)&&(C=C.toInt()),C&=63,C===0)return this;var D=this.high;if(C<32){var O=this.low;return a(O>>>C|D<<32-C,D>>>C,this.unsigned)}else return C===32?a(D,0,this.unsigned):a(D>>>C-32,0,this.unsigned)},k.shru=k.shiftRightUnsigned,k.shr_u=k.shiftRightUnsigned,k.toSigned=function(){return this.unsigned?a(this.low,this.high,!1):this},k.toUnsigned=function(){return this.unsigned?this:a(this.low,this.high,!0)},k.toBytes=function(C){return C?this.toBytesLE():this.toBytesBE()},k.toBytesLE=function(){var C=this.high,D=this.low;return[D&255,D>>>8&255,D>>>16&255,D>>>24,C&255,C>>>8&255,C>>>16&255,C>>>24]},k.toBytesBE=function(){var C=this.high,D=this.low;return[C>>>24,C>>>16&255,C>>>8&255,C&255,D>>>24,D>>>16&255,D>>>8&255,D&255]},t.fromBytes=function(C,D,O){return O?t.fromBytesLE(C,D):t.fromBytesBE(C,D)},t.fromBytesLE=function(C,D){return new t(C[0]|C[1]<<8|C[2]<<16|C[3]<<24,C[4]|C[5]<<8|C[6]<<16|C[7]<<24,D)},t.fromBytesBE=function(C,D){return new t(C[4]<<24|C[5]<<16|C[6]<<8|C[7],C[0]<<24|C[1]<<16|C[2]<<8|C[3],D)},Qc}var fg=O$();const mg=h$(fg),L$=p$({__proto__:null,default:mg},[fg]);const Xs=mg||L$;function Gi(n){return Xs.fromString(n,!0,16)}const gg=Gi("c3a5c85c97cb3127"),Ks=Gi("b492b66fbe98f273"),De=Gi("9ae16a3b2f90404f");function xu(n){return n.xor(n.shru(47))}function xg(n,t,e){const s=n.slice(t,t+e);return Xs.fromBytes(Array.from(s),!0,!0)}function Bt(n,t){return xg(n,t,8)}function uf(n,t){return xg(n,t,4)}function de(n,t){return t===0?n:n.shru(t).or(n.shl(64-t))}function vs(n,t,e=Gi("9ddfea08eb382d69")){let s=n.xor(t).mul(e);s=s.xor(s.shru(47));let o=t.xor(s).mul(e);return o=o.xor(o.shru(47)),o=o.mul(e),o}function M$(n,t,e,s,o,r){o=o.add(n),r=de(r.add(o).add(s),21);const i=o;return o=o.add(t),o=o.add(e),r=r.add(de(o,44)),[o.add(s),r.add(i)]}function fa(n,t,e,s){return M$(Bt(n,t),Bt(n,t+8),Bt(n,t+16),Bt(n,t+24),e,s)}function P$(n,t=n.length){if(t>=8){const e=De.add(t*2),s=Bt(n,0).add(De),o=Bt(n,t-8),r=de(o,37).mul(e).add(s),i=de(s,25).add(o).mul(e);return vs(r,i,e)}if(t>=4){const e=De.add(t*2),s=uf(n,0);return vs(s.shl(3).add(t),uf(n,t-4),e)}if(t>0){const e=n[0],s=n[t>>1],o=n[t-1],r=e+(s<<8),i=t+(o<<2);return xu(De.mul(r).xor(gg.mul(i))).mul(De)}return De}function z$(n,t=n.length){const e=De.add(t*2),s=Bt(n,0).mul(Ks),o=Bt(n,8),r=Bt(n,t-8).mul(e),i=Bt(n,t-16).mul(De);return vs(de(s.add(o),43).add(de(r,30)).add(i),s.add(de(o.add(De),18)).add(r),e)}function B$(n,t=n.length){const e=De.add(t*2),s=Bt(n,0).mul(De),o=Bt(n,8),r=Bt(n,t-8).mul(e),i=Bt(n,t-16).mul(De),a=de(s.add(o),43).add(de(r,30)).add(i),l=vs(a,s.add(de(o.add(De),18)).add(r),e),c=Bt(n,16).mul(e),u=Bt(n,24),h=a.add(Bt(n,t-32)).mul(e),d=l.add(Bt(n,t-24)).mul(e);return vs(de(c.add(u),43).add(de(h,30)).add(d),c.add(de(u.add(s),18)).add(h),e)}function bg(n,t=n.length){const e=Xs.fromNumber(81,!0);if(t<=32)return t<=16?P$(n,t):z$(n,t);if(t<=64)return B$(n,t);let s=e,o=e.mul(Ks).add(113),r=xu(o.mul(De).add(113)).mul(De),i=[Xs.UZERO,Xs.UZERO],a=[Xs.UZERO,Xs.UZERO];s=s.mul(De).add(Bt(n,0));let l=0;const c=(t-1>>6)*64,u=c+(t-1&63)-63;do s=de(s.add(o).add(i[0]).add(Bt(n,l+8)),37).mul(Ks),o=de(o.add(i[1]).add(Bt(n,l+48)),42).mul(Ks),s=s.xor(a[1]),o=o.add(i[0]).add(Bt(n,l+40)),r=de(r.add(a[0]),33).mul(Ks),i=fa(n,l,i[1].mul(Ks),s.add(a[0])),a=fa(n,l+32,r.add(a[1]),o.add(Bt(n,l+16))),[r,s]=[s,r],l+=64;while(l!==c);const h=Ks.add(r.and(255).shl(1));return l=u,a[0]=a[0].add(t-1&63),i[0]=i[0].add(a[0]),a[0]=a[0].add(i[0]),s=de(s.add(o).add(i[0]).add(Bt(n,l+8)),37).mul(h),o=de(o.add(i[1]).add(Bt(n,l+48)),42).mul(h),s=s.xor(a[1].mul(9)),o=o.add(i[0].mul(9).add(Bt(n,l+40))),r=de(r.add(a[0]),33).mul(h),i=fa(n,l,i[1].mul(h),s.add(a[0])),a=fa(n,l+32,r.add(a[1]),o.add(Bt(n,l+16))),[r,s]=[s,r],vs(vs(i[0],a[0],h).add(xu(o).mul(gg)).add(r),vs(i[1],a[1],h).add(s),h)}function as(n,t){return t==="string"?es(n):Ls([n],t)}function V$(n,t){return n instanceof Float32Array&&t==="float32"||n instanceof Int32Array&&t==="int32"||n instanceof Uint8Array&&t==="bool"}function Ls(n,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(n)&&(n=Es(n)),B().getBool("DEBUG")&&qm(n,t),V$(n,t))return n;if(t==null||t==="float32"||t==="complex64")return new Float32Array(n);if(t==="int32")return new Int32Array(n);if(t==="bool"){const e=new Uint8Array(n.length);for(let s=0;s<e.length;++s)Math.round(n[s])!==0&&(e[s]=1);return e}else throw new Error(`Unknown data type ${t}`)}function xe(){return B().platform.now()}function W$(n,t){return B().platform.fetch(n,t)}function es(n,t="utf-8"){return t=t||"utf-8",B().platform.encode(n,t)}function ns(n,t="utf-8"){return t=t||"utf-8",B().platform.decode(n,t)}function je(n){return B().platform.isTypedArray!=null?B().platform.isTypedArray(n):pg(n)}function Es(n,t=[],e=!1){if(t==null&&(t=[]),typeof n=="boolean"||typeof n=="number"||typeof n=="string"||Kr(n)||n==null||je(n)&&e)t.push(n);else if(Array.isArray(n)||je(n))for(let s=0;s<n.length;++s)Es(n[s],t,e);else{let s=-1;for(const o of Object.keys(n))/^([1-9]+[0-9]*|0)$/.test(o)&&(s=Math.max(s,Number(o)));for(let o=0;o<=s;o++)Es(n[o],t,e)}return t}const $K=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:Tt,arraysEqualWithNull:w$,assert:I,assertNonNegativeIntegerDimensions:Fn,assertNonNull:Ju,assertShapesMatch:Oe,bytesFromStringArray:Km,bytesPerElement:Ar,checkConversionForErrors:qm,clamp:Ns,computeStrides:ut,convertBackendValuesAndArrayBuffer:Zm,createScalarValue:as,createShuffledIndices:C$,decodeString:ns,distSquared:b$,encodeString:es,fetch:W$,fingerPrint64:bg,flatten:Es,getArrayFromDType:Jt,getTypedArrayFromDType:we,hasEncodingLoss:th,hexToLong:Gi,indexToLoc:ko,inferDtype:vo,inferFromImplicitShape:Qu,isBoolean:Xm,isFunction:Ts,isInt:oo,isNumber:Ba,isPromise:Kr,isScalarShape:y$,isString:Qo,isTypedArray:je,isValidDtype:jm,locToIndex:Sn,makeOnesTypedArray:ml,makeZerosNestedTypedArray:eh,makeZerosTypedArray:$e,nearestDivisor:Va,nearestLargerEven:fl,now:xe,parseAxisParam:$t,randUniform:x$,repeatedTry:pu,rightPad:to,shuffle:Zu,shuffleCombo:g$,sizeFromShape:U,sizeToSquarishShape:za,squeezeShape:is,sum:Hm,swap:Jn,tanh:$$,toNestedArray:pn,toTypedArray:Ls},Symbol.toStringTag,{value:"Module"}));class U${constructor(t,e){this.backendTimer=t,this.logger=e,e==null&&(this.logger=new H$)}profileKernel(t,e,s){let o;const r=()=>{o=s()};let i;const a=xe();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(r);else{r();for(const c of o)c.dataSync();i=Promise.resolve({kernelMs:xe()-a})}if(B().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let c=0;c<o.length;c++){const u=o[c];u.data().then(h=>{G$(h,u.dtype,t)})}return{kernelName:t,outputs:o,inputs:e,timeMs:i.then(c=>c.kernelMs),extraInfo:i.then(c=>c.getExtraProfileInfo!=null?c.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:e,outputs:s,timeMs:o,inputs:r,extraInfo:i}=t;s.forEach(a=>{Promise.all([a.data(),o,i]).then(l=>{this.logger.logKernelProfile(e,a,l[0],l[1],r,l[2])})})}}function G$(n,t,e){if(t!=="float32")return!1;for(let s=0;s<n.length;s++){const o=n[s];if(isNaN(o)||!isFinite(o))return console.warn(`Found ${o} in the result of '${e}'`),!0}return!1}class H${logKernelProfile(t,e,s,o,r,i){const a=typeof o=="number"?to(`${o}ms`,9):o.error,l=to(t,25),c=e.rank,u=e.size,h=to(e.shape.toString(),14);let d="";for(const p in r){const f=r[p];if(f!=null){const m=f.shape||e.shape,g=m.length;d+=`${p}: ${g}D ${g>0?m:""} `}}console.log(`%c${l}	%c${a}	%c${c}D ${h}	%c${u}	%c${d}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}function q$(n,t,e){const s={},o={};for(let l=0;l<t.length;l++)s[t[l].id]=!0;for(let l=0;l<n.length;l++){const c=n[l],u=c.inputs;for(const h in u){const d=u[h];let p=!1;for(let f=0;f<t.length;f++)if(s[d.id]){c.outputs.forEach(m=>s[m.id]=!0),p=!0,o[c.id]=!0;break}if(p)break}}const r={};r[e.id]=!0;const i={};for(let l=n.length-1;l>=0;l--){const c=n[l],u=c.inputs;for(let h=0;h<c.outputs.length;h++)if(r[c.outputs[h].id]){for(const d in u)r[u[d].id]=!0,i[c.id]=!0;break}}const a=[];for(let l=0;l<n.length;l++){const c=n[l];if(o[c.id]&&i[c.id]){const u={};for(const d in c.inputs){const p=c.inputs[d];s[p.id]&&(u[d]=p)}const h=Object.assign({},c);h.inputs=u,h.outputs=c.outputs,a.push(h)}}return a}function j$(n,t,e,s){for(let o=t.length-1;o>=0;o--){const r=t[o],i=[];if(r.outputs.forEach(l=>{const c=n[l.id];c!=null?i.push(c):i.push(null)}),r.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${r.kernelName}.`);const a=r.gradient(i);for(const l in r.inputs){if(!(l in a))throw new Error(`Cannot backprop through input ${l}. Available gradients found: ${Object.keys(a)}.`);const c=e(()=>a[l]());if(c.dtype!=="float32")throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input ${l} must have 'float32' dtype, but has '${c.dtype}'`);const u=r.inputs[l];if(!Tt(c.shape,u.shape))throw new Error(`Error in gradient for op ${r.kernelName}. The gradient of input '${l}' has shape '${c.shape}', which does not match the shape of the input '${u.shape}'`);if(n[u.id]==null)n[u.id]=c;else{const h=n[u.id];n[u.id]=s(h,c),h.dispose()}}}}const hf=20,gr=3,tu=7;function K$(n,t,e,s){const o=ut(t),r=X$(n,t,e,o),i=t.length,a=va(n,t,e,o,r),l=["Tensor"];return s&&(l.push(`  dtype: ${e}`),l.push(`  rank: ${i}`),l.push(`  shape: [${t}]`),l.push("  values:")),l.push(a.map(c=>"    "+c).join(`
`)),l.join(`
`)}function X$(n,t,e,s){const o=U(t),r=s[s.length-1],i=new Array(r).fill(0),a=t.length,l=e==="complex64"?$r(n):n;if(a>1)for(let c=0;c<o/r;c++){const u=c*r;for(let h=0;h<r;h++)i[h]=Math.max(i[h],wr(l[u+h],0,e).length)}return i}function wr(n,t,e){let s;return Array.isArray(n)?s=`${parseFloat(n[0].toFixed(tu))} + ${parseFloat(n[1].toFixed(tu))}j`:Qo(n)?s=`'${n}'`:e==="bool"?s=yg(n):s=parseFloat(n.toFixed(tu)).toString(),to(s,t)}function yg(n){return n===0?"false":"true"}function va(n,t,e,s,o,r=!0){const i=e==="complex64"?2:1,a=t[0],l=t.length;if(l===0){if(e==="complex64"){const m=$r(n);return[wr(m[0],0,e)]}return e==="bool"?[yg(n[0])]:[n[0].toString()]}if(l===1){if(a>hf){const g=gr*i;let x=Array.from(n.slice(0,g)),b=Array.from(n.slice((a-gr)*i,a*i));return e==="complex64"&&(x=$r(x),b=$r(b)),["["+x.map((w,y)=>wr(w,o[y],e)).join(", ")+", ..., "+b.map((w,y)=>wr(w,o[a-gr+y],e)).join(", ")+"]"]}return["["+(e==="complex64"?$r(n):Array.from(n)).map((g,x)=>wr(g,o[x],e)).join(", ")+"]"]}const c=t.slice(1),u=s.slice(1),h=s[0]*i,d=[];if(a>hf){for(let m=0;m<gr;m++){const g=m*h,x=g+h;d.push(...va(n.slice(g,x),c,e,u,o,!1))}d.push("...");for(let m=a-gr;m<a;m++){const g=m*h,x=g+h;d.push(...va(n.slice(g,x),c,e,u,o,m===a-1))}}else for(let m=0;m<a;m++){const g=m*h,x=g+h;d.push(...va(n.slice(g,x),c,e,u,o,m===a-1))}const p=l===2?",":"";d[0]="["+(a>0?d[0]+p:"");for(let m=1;m<d.length-1;m++)d[m]=" "+d[m]+p;let f=`,
`;for(let m=2;m<l;m++)f+=`
`;return d[d.length-1]=" "+d[d.length-1]+"]"+(r?"":f),d}function $r(n){const t=[];for(let e=0;e<n.length;e+=2)t.push([n[e],n[e+1]]);return t}class ye{constructor(t,e,s){if(this.dtype=e,this.shape=t.slice(),this.size=U(t),s!=null){const o=s.length;I(o===this.size,()=>`Length of values '${o}' does not match the size inferred by the shape '${this.size}'.`)}if(e==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||Jt(e,this.size),this.strides=ut(t)}set(t,...e){e.length===0&&(e=[0]),I(e.length===this.rank,()=>`The number of provided coordinates (${e.length}) must match the rank (${this.rank})`);const s=this.locToIndex(e);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let e=0;for(const o of t){if(o<0||o>=this.shape[e]){const r=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(r)}e++}let s=t[t.length-1];for(let o=0;o<t.length-1;++o)s+=this.strides[o]*t[o];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let e=t[t.length-1];for(let s=0;s<t.length-1;++s)e+=this.strides[s]*t[s];return e}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const e=new Array(this.shape.length);for(let s=0;s<e.length-1;++s)e[s]=Math.floor(t/this.strides[s]),t-=e[s]*this.strides[s];return e[e.length-1]=t,e}get rank(){return this.shape.length}toTensor(){return In().makeTensor(this.values,this.shape,this.dtype)}}let In=null,Mo=null;function Y$(n){In=n}function Z$(n){Mo=n}class Lt{constructor(t,e,s,o){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=e||"float32",this.size=U(t),this.strides=ut(t),this.dataId=s,this.id=o,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return Mo.buffer(this.shape,this.dtype,t)}bufferSync(){return Mo.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return pn(this.shape,t,this.dtype==="complex64")}arraySync(){return pn(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=In().read(this.dataId);if(this.dtype==="string"){const e=await t;try{return e.map(s=>ns(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),In().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=In().readSync(this.dataId);if(this.dtype==="string")try{return t.map(e=>ns(e))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await In().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),In().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return Mo.print(this,t)}clone(){return this.throwIfDisposed(),Mo.clone(this)}toString(t=!1){const e=this.dataSync();return K$(e,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),Mo.cast(this,t)}variable(t=!0,e,s){return this.throwIfDisposed(),In().makeVariable(this,t,e,s)}}Object.defineProperty(Lt,Symbol.hasInstance,{value:n=>!!n&&n.data!=null&&n.dataSync!=null&&n.throwIfDisposed!=null});function G(){return nh("Tensor",()=>Lt)}G();class Ga extends Lt{constructor(t,e,s,o){super(t.shape,t.dtype,t.dataId,o),this.trainable=e,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!Tt(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);In().disposeTensor(this),this.dataId=t.dataId,In().incRef(this,null)}dispose(){In().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(Ga,Symbol.hasInstance,{value:n=>n instanceof Lt&&n.assign!=null&&n.assign instanceof Function});var df;(function(n){n.R0="R0",n.R1="R1",n.R2="R2",n.R3="R3",n.R4="R4",n.R5="R5",n.R6="R6"})(df||(df={}));var bu;(function(n){n.float32="float32",n.int32="int32",n.bool="int32",n.complex64="complex64"})(bu||(bu={}));var yu;(function(n){n.float32="float32",n.int32="int32",n.bool="bool",n.complex64="complex64"})(yu||(yu={}));var wu;(function(n){n.float32="float32",n.int32="float32",n.bool="float32",n.complex64="complex64"})(wu||(wu={}));var $u;(function(n){n.float32="complex64",n.int32="complex64",n.bool="complex64",n.complex64="complex64"})($u||($u={}));const J$={float32:wu,int32:bu,bool:yu,complex64:$u};function Ke(n,t){if(n==="string"||t==="string"){if(n==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${n} with ${t}`)}return J$[n][t]}function Qh(n){return Ke(n,"int32")}function wg(n){return n!=null&&typeof n=="object"&&"texture"in n&&n.texture instanceof WebGLTexture}function $g(n){return typeof GPUBuffer<"u"&&n!=null&&typeof n=="object"&&"buffer"in n&&n.buffer instanceof GPUBuffer}function Qt(n,t){if(n.dtype===t.dtype)return[n,t];const e=Ke(n.dtype,t.dtype);return[n.cast(e),t.cast(e)]}function Q$(n,t){I(n.dtype===t.dtype,()=>`The dtypes of the first(${n.dtype}) and second(${t.dtype}) input must match`)}function xc(n,t){return t.some(e=>e.id===n.id)}function ss(n){const t=[];return Cg(n,t,new Set),t}function Cg(n,t,e){if(n==null)return;if(n instanceof Lt){t.push(n);return}if(!tC(n))return;const s=n;for(const o in s){const r=s[o];e.has(r)||(e.add(r),Cg(r,t,e))}}function tC(n){return Array.isArray(n)||typeof n=="object"}const CK=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:Q$,getTensorsInContainer:ss,isTensorInList:xc,makeTypesMatch:Qt},Symbol.toStringTag,{value:"Module"}));function eu(n){return n.kernelName!=null}class pf{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class qo{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new pf}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const s=t[e];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:e}=this.initializeBackendsAndReturnBest();if(e)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:e}=this.initializeBackend(t);if(e)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,e,s=1){return t in this.registryFactory?(Qe(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:e,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:e,asyncInit:s}=this.initializeBackend(t);if(!(s?await e:e))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new U$(this.backendInstance),!0}setupRegisteredKernels(){gu(this.backendName).forEach(e=>{e.setupFunc!=null&&e.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){gu(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const e=this.registryFactory[t];if(e==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=e.factory();if(s&&!(s instanceof Yu)&&typeof s.then=="function"){const o=++this.pendingBackendInitId,r=s.then(i=>o<this.pendingBackendInitId?!1:(this.registry[t]=i,this.pendingBackendInit=null,!0)).catch(i=>(o<this.pendingBackendInitId||(this.pendingBackendInit=null,Qe(`Initialization of backend ${t} failed`),Qe(i.stack||i.message)),!1));return this.pendingBackendInit=r,{success:r,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return Qe(`Initialization of backend ${t} failed`),Qe(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,e)=>this.registryFactory[e].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let e=0;e<t.length;e++){const s=t[e],{success:o,asyncInit:r}=this.initializeBackend(s);if(r||o)return{name:s,asyncInit:r}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,e){const s=this.state.tensorInfo.get(e),o=s.backend,r=this.readSync(e),i=o.refCount(e);o.disposeData(e,!0),s.backend=t,t.move(e,r,s.shape,s.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,e){let s=null;if(e==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");e=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof e!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let o;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(o),()=>(o=e(),o instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),o))}scopedRun(t,e,s){t();try{const o=s();return e(),o}catch(o){throw e(),o}}nextTensorId(){return qo.nextTensorId++}nextVariableId(){return qo.nextVariableId++}clone(t){const e=R.runKernel(mi,{x:t}),s={x:t},o=i=>({x:()=>{const a="float32",l={x:i},c={dtype:a};return R.runKernel(ni,l,c)}}),r=[];return this.addTapeNode(this.state.activeScope.name,s,[e],o,r,{}),e}runKernel(t,e,s){if(this.backendName==null&&this.backend,!(af(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:e,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,e,s){const o=this.backend.numDataIds();let r=0;s.forEach(l=>{r+=l.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=o-e-r-i;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${t}'`)}runKernelFunc(t){let e,s=[];const o=this.isTapeOn(),r=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let l;const c=eu(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(eu(t)){const{kernelName:f,inputs:m,attrs:g}=t;this.backendName==null&&this.backend;const x=af(f,this.backendName);I(x!=null,()=>`Cannot find registered kernel '${f}' for backend '${this.backendName}'`),a=()=>{const b=this.backend.numDataIds();l=x.kernelFunc({inputs:m,attrs:g,backend:this.backend});const w=Array.isArray(l)?l:[l];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(f,b,w);const y=w.map($=>$.rank!=null?$:this.makeTensorFromTensorInfo($));if(o){const $=this.getTensorsForGradient(f,m,y);s=this.saveTensorsForBackwardMode($)}return y}}else{const{forwardFunc:f}=t,m=g=>{o&&(s=g.map(x=>this.keep(this.clone(x))))};a=()=>{const g=this.backend.numDataIds();l=this.tidy(()=>f(this.backend,m));const x=Array.isArray(l)?l:[l];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,g,x),x}}const{inputs:u,attrs:h}=t,d=eu(t)?null:t.backwardsFunc;let p;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?e=a():(p=this.profiler.profileKernel(c,u,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(p),e=p.outputs)}),o&&this.addTapeNode(c,u,e,d,s,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-r,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(u).map(f=>u[f]!=null?u[f].shape:null),outputShapes:e.map(f=>f.shape),kernelTimeMs:p.timeMs,extraInfo:p.extraInfo}),Array.isArray(l)?e:e[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,e,s){const o=lf(t);if(o!=null){const r=o.inputsToSave||[],i=o.outputsToSave||[];let a;o.saveAllInputs?(I(Array.isArray(e),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(e).map(c=>e[c])):a=r.map(c=>e[c]);const l=s.filter((c,u)=>i[u]);return a.concat(l)}return[]}makeTensor(t,e,s,o){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",o=o||this.backend;let r=t;s==="string"&&Qo(t[0])&&(r=t.map(l=>es(l)));const i=o.write(r,e,s),a=new Lt(e,s,i,this.nextTensorId());if(this.trackTensor(a,o),s==="string"){const l=this.state.tensorInfo.get(i),c=Km(r);this.state.numBytes+=c-l.bytes,l.bytes=c}return a}makeTensorFromDataId(t,e,s,o){s=s||"float32";const r={dataId:t,shape:e,dtype:s};return this.makeTensorFromTensorInfo(r,o)}makeTensorFromTensorInfo(t,e){const{dataId:s,shape:o,dtype:r}=t,i=new Lt(o,r,s,this.nextTensorId());return this.trackTensor(i,e),i}makeVariable(t,e=!0,s,o){s=s||this.nextVariableId().toString(),o!=null&&o!==t.dtype&&(t=t.cast(o));const r=new Ga(t,e,s,this.nextTensorId());if(this.state.registeredVariables[r.name]!=null)throw new Error(`Variable with name ${r.name} was already registered`);return this.state.registeredVariables[r.name]=r,this.incRef(r,this.backend),r}trackTensor(t,e){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*Ar(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:e||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof Ga||this.track(t)}incRef(t,e){this.trackTensor(t,e),this.backend.incRef(t.dataId)}removeDataId(t,e){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===e&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const e=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=e.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*Ar(t.dtype);this.state.numBytes-=s}e.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,e.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const e=this.state.registeredVariables[t];this.disposeVariable(e)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const e=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(o=>o.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-e,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const o of this.state.activeProfile.kernels)o.kernelTimeMs=await o.kernelTimeMs,o.extraInfo=await o.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,e,s,o,r,i){const a={id:this.state.nextTapeNodeId++,kernelName:t,inputs:e,outputs:s,saved:r},l=lf(t);l!=null&&(o=l.gradFunc),o!=null&&(a.gradient=c=>(c=c.map((u,h)=>{if(u==null){const d=s[h],p=$e(d.size,d.dtype);return this.makeTensor(p,d.shape,d.dtype)}return u}),o(c.length>1?c:c[0],r,i))),this.state.activeTape.push(a)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const e={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(e.name=t),this.state.scopeStack.push(e),this.state.activeScope=e}endScope(t){const e=ss(t),s=new Set(e.map(r=>r.id));for(let r=0;r<this.state.activeScope.track.length;r++){const i=this.state.activeScope.track[r];!i.kept&&!s.has(i.id)&&i.dispose()}const o=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],e.forEach(r=>{!r.kept&&r.scopeId===o.id&&this.track(r)})}gradients(t,e,s,o=!1){if(I(e.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const r=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));I(r instanceof Lt,()=>"The result y returned by f() must be a tensor.");const i=q$(this.state.activeTape,e,r);if(!o&&i.length===0&&e.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[r.id]=s??eC(r.shape),j$(a,i,c=>this.tidy(c),nC);const l=e.map(c=>a[c.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(c=>{for(const u of c.saved)u.dispose()}),this.state.activeTape=null),{value:r,grads:l}})}customGrad(t){return I(Ts(t),()=>"The f passed in customGrad(f) must be a function."),(...e)=>{I(e.every(a=>a instanceof Lt),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const o={};e.forEach((a,l)=>{o[l]=a});const r=(a,l)=>(s=t(...e,l),I(s.value instanceof Lt,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),I(Ts(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),i=(a,l)=>{const c=s.gradFunc(a,l),u=Array.isArray(c)?c:[c];I(u.length===e.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),I(u.every(d=>d instanceof Lt),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const h={};return u.forEach((d,p)=>{h[p]=()=>d}),h};return this.runKernelFunc({forwardFunc:r,backwardsFunc:i,inputs:o})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,e){return this.state.tensorInfo.get(t).backend.readToGPU(t,e)}async time(t){const e=xe(),s=await this.backend.time(t);return s.wallMs=xe()-e,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new pf;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}qo.nextTensorId=0;qo.nextVariableId=0;function eC(n){const t=ml(U(n),"float32");return R.makeTensor(t,n,"float32")}function Ig(){const n=Qm();if(n._tfengine==null){const t=new I$(n);n._tfengine=new qo(t)}return N$(n._tfengine.ENV),Y$(()=>n._tfengine),n._tfengine}const R=Ig();function nC(n,t){const e={a:n,b:t};return R.runKernel(tr,e)}function sC(){return typeof navigator<"u"&&navigator!=null}let Cu;function oC(n){Cu=n}function td(n){if(Cu!==void 0)return Cu;if(n||sC()){if(n||(n=navigator),n.product==="ReactNative")return!0;const t=n.userAgent||n.vendor||(typeof window<"u"?window.opera:"");if(!t){const e=n;return e.userAgentData&&e.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function ed(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const IK=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:ed,isMobile:td,mockIsMobile:oC},Symbol.toStringTag,{value:"Module"}));const Ve=B();Ve.registerFlag("DEBUG",()=>!1,n=>{n&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});Ve.registerFlag("IS_BROWSER",()=>ed());Ve.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");Ve.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));Ve.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));Ve.registerFlag("PROD",()=>!1);Ve.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>Ve.getBool("DEBUG"));Ve.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);Ve.registerFlag("IS_TEST",()=>!1);Ve.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>Ve.getBool("DEBUG"));Ve.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);Ve.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);Ve.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);function bc(n,t){let e=n;if(je(n))return t==="string"?[]:[n.length];if(wg(n)){const o=n.channels||"RGBA";return[n.height,n.width*o.length]}else if($g(n))return[n.buffer.size/(t==null?4:Ar(t))];if(!Array.isArray(n))return[];const s=[];for(;Array.isArray(e)||je(e)&&t!=="string";)s.push(e.length),e=e[0];return Array.isArray(n)&&B().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&vg(n,s,[]),s}function vg(n,t,e){if(e=e||[],!Array.isArray(n)&&!je(n)){I(t.length===0,()=>`Element arr[${e.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}I(t.length>0,()=>`Element arr[${e.join("][")}] should be a primitive, but is an array of ${n.length} elements`),I(n.length===t[0],()=>`Element arr[${e.join("][")}] should have ${t[0]} elements, but has ${n.length} elements`);const s=t.slice(1);for(let o=0;o<n.length;++o)vg(n[o],s,e.concat(o))}function ff(n,t,e,s){if(n!=="string_or_numeric"){if(n==null)throw new Error("Expected dtype cannot be null.");if(n!=="numeric"&&n!==t||n==="numeric"&&t==="string")throw new Error(`Argument '${e}' passed to '${s}' must be ${n} tensor, but got ${t} tensor`)}}function E(n,t,e,s="numeric"){if(n instanceof G())return ff(s,n.dtype,t,e),n;let o=vo(n);if(o!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(o=s),ff(s,o,t,e),n==null||!je(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string"){const l=n==null?"null":n.constructor.name;throw new Error(`Argument '${t}' passed to '${e}' must be a Tensor or TensorLike, but got '${l}'`)}const r=bc(n,o);!je(n)&&!Array.isArray(n)&&(n=[n]);const a=o!=="string"?Ls(n,o):Es(n,[],!0);return R.makeTensor(a,r,o)}function nd(n,t,e,s="numeric"){if(!Array.isArray(n))throw new Error(`Argument ${t} passed to ${e} must be a \`Tensor[]\` or \`TensorLike[]\``);return n.map((r,i)=>E(r,`${t}[${i}]`,e,s))}const rC="__op";function _(n){const t=Object.keys(n);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let e=t[0];const s=n[e];e.endsWith("_")&&(e=e.substring(0,e.length-1)),e=e+rC;const o=(...r)=>{R.startScope(e);try{const i=s(...r);return Kr(i)&&console.error("Cannot return a Promise inside of tidy."),R.endScope(i),i}catch(i){throw R.endScope(null),i}};return Object.defineProperty(o,"name",{value:e,configurable:!0}),o}function iC(n,t){const e=E(n,"real","complex"),s=E(t,"imag","complex");Oe(e.shape,s.shape,`real and imag shapes, ${e.shape} and ${s.shape}, must match in call to tf.complex().`);const o={real:e,imag:s};return R.runKernel(uh,o)}const ro=_({complex_:iC});function yc(n,t,e,s){if(s==null)s=vo(n);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if($g(n)||wg(n)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return R.backend.createTensorFromGPUData(n,t||e,s)}if(!je(n)&&!Array.isArray(n)&&typeof n!="number"&&typeof n!="boolean"&&typeof n!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){Fn(t);const o=U(t),r=U(e);I(o===r,()=>`Based on the provided shape, [${t}], the tensor should have ${o} values but has ${r}`);for(let i=0;i<e.length;++i){const a=e[i],l=i===e.length-1?a!==U(t.slice(i)):!0;I(e[i]===t[i]||!l,()=>`Error creating a new Tensor. Inferred shape (${e}) does not match the provided shape (${t}). `)}}return!je(n)&&!Array.isArray(n)&&(n=[n]),t=t||e,n=s!=="string"?Ls(n,s):Es(n,[],!0),R.makeTensor(n,t,s)}function Bo(n,t,e){const s=bc(n,e);return yc(n,t,s,e)}const io={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class ls{static join(t){return new ls(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>je(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let e=0;for(let s=0;s<t.length;s++){const o=t[s];s!==t.length-1&&o.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const r=e+o.byteLength;this.shards.push({buffer:o,start:e,end:r}),e=r}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,e=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,e=isNaN(Number(e))?0:e,t=Math.max(0,t),e=Math.min(this.byteLength,e),e<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const o=e-t,r=new ArrayBuffer(o),i=new Uint8Array(r);let a=0;for(let l=s;l<this.shards.length;l++){const c=this.shards[l],h=t+a-c.start,d=a,f=Math.min(e,c.end)-c.start,m=new Uint8Array(c.buffer,h,f-h);if(i.set(m,d),a+=m.length,e<c.end)break}return r}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function e(o){return t<o.start?-1:t>=o.end?1:0}if(e(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=aC(this.shards,e);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function aC(n,t){let e=0,s=n.length;for(;e<=s;){const o=Math.floor((s-e)/2)+e,r=t(n[o]);if(r===0)return o;r<0?s=o:e=o+1}return-1}function vK(){B().set("PROD",!0)}function kK(){B().set("DEBUG",!0)}function SK(){B().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function NK(n){B().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(n+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function TK(){R.disposeVariables()}function On(){return R}function mf(){return R.memory()}function EK(n){return R.profile(n)}function z(n,t){return R.tidy(n,t)}function xt(n){ss(n).forEach(e=>e.dispose())}function Bn(n){return R.keep(n)}function RK(n){return R.time(n)}function AK(n){return R.setBackend(n)}function DK(){return R.ready()}function lC(){return R.backendName}function FK(n){R.removeBackend(n)}function _K(n){return R.findBackend(n)}function OK(n){return R.findBackendFactory(n)}function kg(n,t,e=1){return R.registerBackend(n,t,e)}function Sg(){return R.backend}function LK(n,t){B().setPlatform(n,t)}const Rs=4;async function gf(n,t){const e=[],s=[],o=Array.isArray(n)?n.map(i=>i.name):Object.keys(n);for(let i=0;i<o.length;++i){const a=o[i],l=Array.isArray(n)?n[i].tensor:n[a];if(l.dtype!=="float32"&&l.dtype!=="int32"&&l.dtype!=="bool"&&l.dtype!=="string"&&l.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${l.dtype}`);const c={name:a,shape:l.shape,dtype:l.dtype};if(l.dtype==="string"){const u=new Promise(async h=>{const d=await l.bytes(),p=d.reduce((g,x)=>g+x.length,0)+Rs*d.length,f=new Uint8Array(p);let m=0;for(let g=0;g<d.length;g++){const x=d[g],b=new Uint8Array(new Uint32Array([x.length]).buffer);f.set(b,m),m+=Rs,f.set(x,m),m+=x.length}h(f)});s.push(u)}else s.push(l.data());t!=null&&(c.group=t),e.push(c)}const r=await Promise.all(s);return{data:hC(r),specs:e}}function Ng(n,t){const e=new ls(n),s={};let o=0;for(const r of t){const i=cC(r,(a,l)=>e.slice(o+a,o+l));s[r.name]=Tg(r,e.slice(o,o+i)),o+=i}return s}function cC(n,t){const e=U(n.shape);let s;if("quantization"in n){const o=n.quantization;s=io[o.dtype]}else if(n.dtype==="string"){let o=0;for(let r=0;r<e;r++)o+=Rs+new Uint32Array(t(o,o+Rs))[0];return o}else s=io[n.dtype];return e*s}async function uC(n,t){const e=U(n.shape);let s;if("quantization"in n){const o=n.quantization;s=io[o.dtype]}else if(n.dtype==="string"){let o=0;for(let r=0;r<e;r++)o+=Rs+new Uint32Array(await t(o,o+Rs))[0];return o}else s=io[n.dtype];return e*s}function Tg(n,t){const e=n.name,s=n.dtype,o=n.shape,r=U(o);let i,a=0;if("quantization"in n){const l=n.quantization;if(l.dtype==="uint8"||l.dtype==="uint16"){if(!("min"in l&&"scale"in l))throw new Error(`Weight ${n.name} with quantization ${l.dtype} doesn't have corresponding metadata min and scale.`)}else if(l.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${n.name} is quantized with ${l.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${n.name} has unknown quantization dtype ${l.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const c=io[l.dtype],u=l.dtype==="uint8"?new Uint8Array(t):new Uint16Array(t);if(s==="float32")if(l.dtype==="uint8"||l.dtype==="uint16"){i=new Float32Array(u.length);for(let h=0;h<u.length;h++){const d=u[h];i[h]=d*l.scale+l.min}}else if(l.dtype==="float16")i=$C()(u);else throw new Error(`Unsupported quantization type ${l.dtype} for weight type float32.`);else if(s==="int32"){if(l.dtype!=="uint8"&&l.dtype!=="uint16")throw new Error(`Unsupported quantization type ${l.dtype} for weight type int32.`);i=new Int32Array(u.length);for(let h=0;h<u.length;h++){const d=u[h];i[h]=Math.round(d*l.scale+l.min)}}else throw new Error(`Unsupported dtype in weight '${e}': ${s}`);a+=r*c}else if(s==="string"){const l=U(n.shape);i=[];for(let c=0;c<l;c++){const u=new Uint32Array(t.slice(a,a+Rs))[0];a+=Rs;const h=new Uint8Array(t.slice(a,a+u));i.push(h),a+=u}}else{const l=io[s];if(s==="float32")i=new Float32Array(t);else if(s==="int32")i=new Int32Array(t);else if(s==="bool")i=new Uint8Array(t);else if(s==="complex64"){i=new Float32Array(t);const c=new Float32Array(i.length/2),u=new Float32Array(i.length/2);for(let f=0;f<c.length;f++)c[f]=i[f*2],u[f]=i[f*2+1];const h=Bo(c,o,"float32"),d=Bo(u,o,"float32"),p=ro(h,d);return h.dispose(),d.dispose(),p}else throw new Error(`Unsupported dtype in weight '${e}': ${s}`);a+=r*l}return Bo(i,o,s)}async function xf(n,t,e){let s=new Uint8Array(t);for(;s.byteLength<e;){const{done:o,value:r}=await n.read();if(o&&r==null){const a=e-s.byteLength;throw new Error(`Reader is done but ${a} bytes are still expected`)}const i=new Uint8Array(s.length+r.byteLength);i.set(s,0),i.set(new Uint8Array(r),s.length),s=i}return s.buffer}async function MK(n,t){const e={},s=n.getReader();let o=new ArrayBuffer(0);for(const r of t){const i=await uC(r,async(c,u)=>(o=await xf(s,o,u),o.slice(c,u)));o=await xf(s,o,i);const a=o.slice(0,i);o=o.slice(i);const l=Tg(r,a);if(e[r.name]=l,lC()==="webgpu"){const c=Sg();"uploadToGPU"in c&&U(l.shape)>=B().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&c.uploadToGPU(l.dataId)}}return e}function hC(n){if(n===null)throw new Error(`Invalid input value: ${JSON.stringify(n)}`);let t=0;const e=[];n.forEach(r=>{if(t+=r.byteLength,e.push(r.byteLength===r.buffer.byteLength?r:new r.constructor(r)),!(r instanceof Float32Array||r instanceof Int32Array||r instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${r.constructor.name}`)});const s=new Uint8Array(t);let o=0;return e.forEach(r=>{s.set(new Uint8Array(r.buffer),o),o+=r.byteLength}),s.buffer}const sd=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function bf(n){return sd?Buffer.byteLength(n,"utf8"):new Blob([n]).size}function dC(n){if(sd)return Buffer.from(n).toString("base64");const t=new Uint8Array(n);let e="";for(let s=0,o=t.length;s<o;s++)e+=String.fromCharCode(t[s]);return btoa(e)}function pC(n){if(sd){const s=Buffer.from(n,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(n),e=new Uint8Array(t.length);for(let s=0;s<t.length;++s)e.set([t.charCodeAt(s)],s);return e.buffer}function fC(n){return ls.join(n)}function PK(n){for(n=n.trim();n.endsWith("/");)n=n.slice(0,n.length-1);const e=n.split("/");return e[e.length-1]}function mC(n,t){const e={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy,weightsManifest:t};return n.signature!=null&&(e.signature=n.signature),n.userDefinedMetadata!=null&&(e.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(e.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(e.initializerSignature=n.initializerSignature),n.trainingConfig!=null&&(e.trainingConfig=n.trainingConfig),e}function gC(n,t,e){const s={modelTopology:n.modelTopology,format:n.format,generatedBy:n.generatedBy,convertedBy:n.convertedBy};if(n.trainingConfig!=null&&(s.trainingConfig=n.trainingConfig),n.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!e)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=e}return n.signature!=null&&(s.signature=n.signature),n.userDefinedMetadata!=null&&(s.userDefinedMetadata=n.userDefinedMetadata),n.modelInitializer!=null&&(s.modelInitializer=n.modelInitializer),n.initializerSignature!=null&&(s.initializerSignature=n.initializerSignature),s}async function xC(n,t){let e,s;return n.weightsManifest!=null&&([e,s]=await t(n.weightsManifest)),gC(n,e,s)}function od(n){if(n.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:n.modelTopology==null?0:bf(JSON.stringify(n.modelTopology)),weightSpecsBytes:n.weightSpecs==null?0:bf(JSON.stringify(n.weightSpecs)),weightDataBytes:n.weightData==null?0:new ls(n.weightData).byteLength}}function yf(n){const t=[];for(const e of n)t.push(...e.weights);return t}function bC(){const n=e=>{let s=e<<13,o=0;for(;(s&8388608)===0;)o-=8388608,s<<=1;return s&=-8388609,o+=947912704,s|o},t=new Uint32Array(2048);t[0]=0;for(let e=1;e<1024;e++)t[e]=n(e);for(let e=1024;e<2048;e++)t[e]=939524096+(e-1024<<13);return t}function yC(){const n=new Uint32Array(64);n[0]=0,n[31]=1199570944,n[32]=2147483648,n[63]=3347054592;for(let t=1;t<31;t++)n[t]=t<<23;for(let t=33;t<63;t++)n[t]=2147483648+(t-32<<23);return n}function wC(){const n=new Uint32Array(64);for(let t=0;t<64;t++)n[t]=1024;return n[0]=n[32]=0,n}function $C(){const n=bC(),t=yC(),e=wC();return s=>{const o=new ArrayBuffer(4*s.length),r=new Uint32Array(o);for(let i=0;i<s.length;i++){const a=s[i],l=n[e[a>>10]+(a&1023)]+t[a>>10];r[i]=l}return new Float32Array(o)}}class Zt{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return Zt.instance==null&&(Zt.instance=new Zt),Zt.instance}static registerSaveRouter(t){Zt.getInstance().saveRouters.push(t)}static registerLoadRouter(t){Zt.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return Zt.getHandlers(t,"save")}static getLoadHandlers(t,e){return Zt.getHandlers(t,"load",e)}static getHandlers(t,e,s){const o=[];return(e==="load"?Zt.getInstance().loadRouters:Zt.getInstance().saveRouters).forEach(i=>{const a=i(t,s);a!==null&&o.push(a)}),o}}const zK=n=>Zt.registerSaveRouter(n),BK=n=>Zt.registerLoadRouter(n),CC=n=>Zt.getSaveHandlers(n),IC=(n,t)=>Zt.getLoadHandlers(n,t);const Iu="tensorflowjs",vu=1,Js="models_store",ws="model_info_store";function Eg(){if(!B().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const n=typeof window>"u"?self:window,t=n.indexedDB||n.mozIndexedDB||n.webkitIndexedDB||n.msIndexedDB||n.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function ku(n){const t=n.result;t.createObjectStore(Js,{keyPath:"modelPath"}),t.createObjectStore(ws,{keyPath:"modelPath"})}class ao{constructor(t){if(this.indexedDB=Eg(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,e){return new Promise((s,o)=>{const r=this.indexedDB.open(Iu,vu);r.onupgradeneeded=()=>ku(r),r.onsuccess=()=>{const i=r.result;if(e==null){const a=i.transaction(Js,"readonly"),c=a.objectStore(Js).get(this.modelPath);c.onsuccess=()=>{if(c.result==null)return i.close(),o(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(c.result.modelArtifacts)},c.onerror=u=>(i.close(),o(c.error)),a.oncomplete=()=>i.close()}else{e.weightData=ls.join(e.weightData);const a=od(e),l=i.transaction(ws,"readwrite");let c=l.objectStore(ws),u;try{u=c.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(d){return o(d)}let h;u.onsuccess=()=>{h=i.transaction(Js,"readwrite");const d=h.objectStore(Js);let p;try{p=d.put({modelPath:this.modelPath,modelArtifacts:e,modelArtifactsInfo:a})}catch(f){return o(f)}p.onsuccess=()=>s({modelArtifactsInfo:a}),p.onerror=f=>{c=l.objectStore(ws);const m=c.delete(this.modelPath);m.onsuccess=()=>(i.close(),o(p.error)),m.onerror=g=>(i.close(),o(p.error))}},u.onerror=d=>(i.close(),o(u.error)),l.oncomplete=()=>{h==null?i.close():h.oncomplete=()=>i.close()}}},r.onerror=i=>o(r.error)})}}ao.URL_SCHEME="indexeddb://";const Rg=n=>B().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(ao.URL_SCHEME)?vC(n.slice(ao.URL_SCHEME.length)):null;Zt.registerSaveRouter(Rg);Zt.registerLoadRouter(Rg);function vC(n){return new ao(n)}function kC(n){return n.startsWith(ao.URL_SCHEME)?n.slice(ao.URL_SCHEME.length):n}class SC{constructor(){this.indexedDB=Eg()}async listModels(){return new Promise((t,e)=>{const s=this.indexedDB.open(Iu,vu);s.onupgradeneeded=()=>ku(s),s.onsuccess=()=>{const o=s.result,r=o.transaction(ws,"readonly"),a=r.objectStore(ws).getAll();a.onsuccess=()=>{const l={};for(const c of a.result)l[c.modelPath]=c.modelArtifactsInfo;t(l)},a.onerror=l=>(o.close(),e(a.error)),r.oncomplete=()=>o.close()},s.onerror=o=>e(s.error)})}async removeModel(t){return t=kC(t),new Promise((e,s)=>{const o=this.indexedDB.open(Iu,vu);o.onupgradeneeded=()=>ku(o),o.onsuccess=()=>{const r=o.result,i=r.transaction(ws,"readwrite"),a=i.objectStore(ws),l=a.get(t);let c;l.onsuccess=()=>{if(l.result==null)return r.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const u=a.delete(t),h=()=>{c=r.transaction(Js,"readwrite");const p=c.objectStore(Js).delete(t);p.onsuccess=()=>e(l.result.modelArtifactsInfo),p.onerror=f=>s(l.error)};u.onsuccess=h,u.onerror=d=>(h(),r.close(),s(l.error))}},l.onerror=u=>(r.close(),s(l.error)),i.oncomplete=()=>{c==null?r.close():c.oncomplete=()=>r.close()}},o.onerror=r=>s(o.error)})}}const Qn="/",Po="tensorflowjs_models",Ag="info",NC="model_topology",TC="weight_specs",EC="weight_data",RC="model_metadata";function Dg(n){return{info:[Po,n,Ag].join(Qn),topology:[Po,n,NC].join(Qn),weightSpecs:[Po,n,TC].join(Qn),weightData:[Po,n,EC].join(Qn),modelMetadata:[Po,n,RC].join(Qn)}}function Fg(n){for(const t of Object.values(n))window.localStorage.removeItem(t)}function AC(n){const t=n.split(Qn);if(t.length<3)throw new Error(`Invalid key format: ${n}`);return t.slice(1,t.length-1).join(Qn)}function DC(n){return n.startsWith(lo.URL_SCHEME)?n.slice(lo.URL_SCHEME.length):n}class lo{constructor(t){if(!B().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=Dg(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const e=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),o=od(t),r=ls.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(o)),this.LS.setItem(this.keys.topology,e),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,dC(r));const i={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(i)),{modelArtifactsInfo:o}}catch{throw Fg(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${o.modelTopologyBytes}, weightSpecsBytes=${o.weightSpecsBytes}, weightDataBytes=${o.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const e={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);e.modelTopology=s;const o=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(o==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);e.weightSpecs=o;const r=this.LS.getItem(this.keys.modelMetadata);if(r!=null){const a=JSON.parse(r);e.format=a.format,e.generatedBy=a.generatedBy,e.convertedBy=a.convertedBy,a.signature!=null&&(e.signature=a.signature),a.userDefinedMetadata!=null&&(e.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(e.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(e.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(e.trainingConfig=a.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return e.weightData=pC(i),e}}lo.URL_SCHEME="localstorage://";const _g=n=>B().getBool("IS_BROWSER")&&!Array.isArray(n)&&n.startsWith(lo.URL_SCHEME)?FC(n.slice(lo.URL_SCHEME.length)):null;Zt.registerSaveRouter(_g);Zt.registerLoadRouter(_g);function FC(n){return new lo(n)}class _C{constructor(){I(B().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),I(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},e=Po+Qn,s=Qn+Ag;for(let o=0;o<this.LS.length;++o){const r=this.LS.key(o);if(r.startsWith(e)&&r.endsWith(s)){const i=AC(r);t[i]=JSON.parse(this.LS.getItem(r))}}return t}async removeModel(t){t=DC(t);const e=Dg(t);if(this.LS.getItem(e.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(e.info));return Fg(e),s}}const Vo="://";class Ae{constructor(){this.managers={}}static getInstance(){return Ae.instance==null&&(Ae.instance=new Ae),Ae.instance}static registerManager(t,e){I(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(Vo)&&(t=t.slice(0,t.indexOf(Vo))),I(t.length>0,()=>"scheme must not be an empty string.");const s=Ae.getInstance();I(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=e}static getManager(t){const e=Ae.getInstance().managers[t];if(e==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return e}static getSchemes(){return Object.keys(Ae.getInstance().managers)}}function ka(n){if(n.indexOf(Vo)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${Ae.getSchemes().join(",")}`);return{scheme:n.split(Vo)[0],path:n.split(Vo)[1]}}async function Og(n,t,e=!1){I(n!==t,()=>`Old path and new path are the same: '${n}'`);const s=Zt.getLoadHandlers(n);I(s.length>0,()=>`Copying failed because no load handler is found for source URL ${n}.`),I(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${n}.`);const o=s[0],r=Zt.getSaveHandlers(t);I(r.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),I(r.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const i=r[0],a=ka(n).scheme,l=ka(n).path,c=a===ka(n).scheme,u=await o.load();e&&c&&await Ae.getManager(a).removeModel(l);const h=await i.save(u);return e&&!c&&await Ae.getManager(a).removeModel(l),h.modelArtifactsInfo}async function VK(){const n=Ae.getSchemes(),t={};for(const e of n){const s=await Ae.getManager(e).listModels();for(const o in s){const r=e+Vo+o;t[r]=s[o]}}return t}async function WK(n){const t=ka(n);return Ae.getManager(t.scheme).removeModel(t.path)}async function UK(n,t){return Og(n,t,!1)}async function GK(n,t){return Og(n,t,!0)}class OC{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,e){return fetch(t,e)}now(){return performance.now()}encode(t,e){if(e!=="utf-8"&&e!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${e}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,e){return new TextDecoder(e).decode(t)}setTimeoutCustom(t,e){if(typeof window>"u"||!B().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,e);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},e),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const o=this.functionRefs[s.data.index];o(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return pg(t)}}if(B().get("IS_BROWSER")){B().setPlatform("browser",new OC);try{Ae.registerManager(lo.URL_SCHEME,new _C)}catch{}try{Ae.registerManager(ao.URL_SCHEME,new SC)}catch{}}const LC={importFetch:()=>require("node-fetch")};let nu;class MC{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,e){return B().global.fetch!=null?B().global.fetch(t,e):(nu==null&&(nu=LC.importFetch()),nu(t,e))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,e){if(e!=="utf-8"&&e!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${e}`);return this.textEncoder.encode(t)}decode(t,e){return t.length===0?"":new this.util.TextDecoder(e).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}B().get("IS_NODE")&&!B().get("IS_BROWSER")&&B().setPlatform("node",new MC);function Ct(n,t="float32",e){return t=t||"float32",Fn(n),new ye(n,t,e)}function PC(n,t){const e=E(n,"x","cast");if(!jm(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&e.dtype!=="string"||t!=="string"&&e.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:e},o={dtype:t};return R.runKernel(ni,s,o)}const st=_({cast_:PC});function zC(n){const e={x:E(n,"x","clone","string_or_numeric")};return R.runKernel(mi,e)}const eo=_({clone_:zC});function BC(n,t=!1){console.log(n.toString(t))}Ig();const VC={buffer:Ct,cast:st,clone:eo,print:BC};Z$(VC);function WC(n,t){let e=E(n,"a","add"),s=E(t,"b","add");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(tr,o)}const J=_({add_:WC});function UC(n,t){let e=E(n,"a","floorDiv"),s=E(t,"b","floorDiv");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(pi,o)}const Lg=_({floorDiv_:UC});function GC(n,t){let e=E(n,"a","div"),s=E(t,"b","div");if([e,s]=Qt(e,s),e.dtype==="int32"&&s.dtype==="int32")return Lg(e,s);const o={a:e,b:s},r={};return R.runKernel(ai,o,r)}const ct=_({div_:GC});function HC(n,t){let e=E(n,"a","mul"),s=E(t,"b","mul");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(vi,o)}const A=_({mul_:HC});function qC(n){const t=E(n,"x","abs");if(t.dtype==="complex64"){const e={x:t};return R.runKernel(Il,e)}else{const e={x:t};return R.runKernel(gl,e)}}const he=_({abs_:qC});function jC(n){const e={x:E(n,"x","acos")};return R.runKernel(Xr,e)}const KC=_({acos_:jC});function XC(n){const e={x:E(n,"x","acosh")};return R.runKernel(Yr,e)}const YC=_({acosh_:XC});function ZC(n,t=null,e=!1){const o={x:E(n,"x","all","bool")},r={axis:t,keepDims:e};return R.runKernel(oh,o,r)}const Mg=_({all_:ZC});function JC(n,t=null,e=!1){const o={x:E(n,"x","any","bool")},r={axis:t,keepDims:e};return R.runKernel(rh,o,r)}const Su=_({any_:JC});function QC(n,t=0){const s={x:E(n,"x","argMax")},o={axis:t};return R.runKernel(xl,s,o)}const Fr=_({argMax_:QC});function tI(n,t=0){const s={x:E(n,"x","argMin")},o={axis:t};return R.runKernel(bl,s,o)}const eI=_({argMin_:tI});function nI(n){const e={x:E(n,"x","asin")};return R.runKernel(Zr,e)}const sI=_({asin_:nI});function oI(n){const e={x:E(n,"x","asinh")};return R.runKernel(Jr,e)}const rI=_({asinh_:oI});function iI(n){const e={x:E(n,"x","atan")};return R.runKernel(Qr,e)}const aI=_({atan_:iI});function lI(n,t){let e=E(n,"a","atan2"),s=E(t,"b","atan2");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(ei,o)}const cI=_({atan2_:lI});function uI(n){const e={x:E(n,"x","atanh")};return R.runKernel(ti,e)}const hI=_({atanh_:uI});function Hi(n,t,e,s,o="NHWC",r){const i=n[3],a=[...t,i],l=us(o);return Ce(n,a,e,r,s,null,null,l)}function xn(n,t,e,s,o,r,i="channelsLast"){const[a,l]=_r(t);let c;if(i==="channelsLast")c=[a,l,n[3],n[3]];else if(i==="channelsFirst")c=[a,l,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return Ce(n,c,e,s,o,r,!1,i)}function cs(n,t,e,s,o,r,i="NDHWC"){const[a,l,c]=Nu(t);let u,h;if(i==="NDHWC")h="channelsLast",u=[a,l,c,n[4],n[4]];else if(i==="NCDHW")h="channelsFirst",u=[a,l,c,n[1],n[1]];else throw new Error(`Unknown dataFormat ${i}`);return Ms(n,u,e,s,o,!1,h,r)}function Ce(n,t,e,s,o,r,i=!1,a="channelsLast"){let[l,c,u,h]=[-1,-1,-1,-1];if(a==="channelsLast")[l,c,u,h]=n;else if(a==="channelsFirst")[l,h,c,u]=n;else throw new Error(`Unknown dataFormat ${a}`);const[d,p,,f]=t,[m,g]=_r(e),[x,b]=_r(s),w=Wo(d,x),y=Wo(p,b),{padInfo:$,outHeight:v,outWidth:N}=fI(o,c,u,m,g,w,y,r,a),T=i?f*h:f;let k;return a==="channelsFirst"?k=[l,T,v,N]:a==="channelsLast"&&(k=[l,v,N,T]),{batchSize:l,dataFormat:a,inHeight:c,inWidth:u,inChannels:h,outHeight:v,outWidth:N,outChannels:T,padInfo:$,strideHeight:m,strideWidth:g,filterHeight:d,filterWidth:p,effectiveFilterHeight:w,effectiveFilterWidth:y,dilationHeight:x,dilationWidth:b,inShape:n,outShape:k,filterShape:t}}function Ms(n,t,e,s,o,r=!1,i="channelsLast",a){let[l,c,u,h,d]=[-1,-1,-1,-1,-1];if(i==="channelsLast")[l,c,u,h,d]=n;else if(i==="channelsFirst")[l,d,c,u,h]=n;else throw new Error(`Unknown dataFormat ${i}`);const[p,f,m,,g]=t,[x,b,w]=Nu(e),[y,$,v]=Nu(s),N=Wo(p,y),T=Wo(f,$),k=Wo(m,v),{padInfo:S,outDepth:C,outHeight:D,outWidth:O}=mI(o,c,u,h,x,b,w,N,T,k,a),P=r?g*d:g;let M;return i==="channelsFirst"?M=[l,P,C,D,O]:i==="channelsLast"&&(M=[l,C,D,O,P]),{batchSize:l,dataFormat:i,inDepth:c,inHeight:u,inWidth:h,inChannels:d,outDepth:C,outHeight:D,outWidth:O,outChannels:P,padInfo:S,strideDepth:x,strideHeight:b,strideWidth:w,filterDepth:p,filterHeight:f,filterWidth:m,effectiveFilterDepth:N,effectiveFilterHeight:T,effectiveFilterWidth:k,dilationDepth:y,dilationHeight:$,dilationWidth:v,inShape:n,outShape:M,filterShape:t}}function dI(n,t,e,s,o){s==null&&(s=rd(n,t,e));const r=n[0],i=n[1],a=Or((r-t+2*s)/e+1,o),l=Or((i-t+2*s)/e+1,o);return[a,l]}function pI(n,t,e,s,o,r){o==null&&(o=rd(n,t[0],s[0]));const i=[0,0,0,e];for(let a=0;a<3;a++)n[a]+2*o>=t[a]&&(i[a]=Or((n[a]-t[a]+2*o)/s[a]+1,r));return i}function rd(n,t,e,s=1){const o=Wo(t,s);return Math.floor((n[0]*(e-1)-e+o)/2)}function _r(n){return typeof n=="number"?[n,n,n]:n.length===2?[n[0],n[1],1]:n}function Nu(n){return typeof n=="number"?[n,n,n]:n}function Wo(n,t){return t<=1?n:n+(n-1)*(t-1)}function fI(n,t,e,s,o,r,i,a,l){let c,u,h;if(typeof n=="number"){c={top:n,bottom:n,left:n,right:n,type:n===0?"VALID":"NUMBER"};const p=dI([t,e],r,s,n,a);u=p[0],h=p[1]}else if(n==="same"){u=Math.ceil(t/s),h=Math.ceil(e/o);const d=Math.max(0,(u-1)*s+r-t),p=Math.max(0,(h-1)*o+i-e),f=Math.floor(d/2),m=d-f,g=Math.floor(p/2),x=p-g;c={top:f,bottom:m,left:g,right:x,type:"SAME"}}else if(n==="valid")c={top:0,bottom:0,left:0,right:0,type:"VALID"},u=Math.ceil((t-r+1)/s),h=Math.ceil((e-i+1)/o);else if(typeof n=="object"){const d=l==="channelsLast"?n[1][0]:n[2][0],p=l==="channelsLast"?n[1][1]:n[2][1],f=l==="channelsLast"?n[2][0]:n[3][0],m=l==="channelsLast"?n[2][1]:n[3][1];c={top:d,bottom:p,left:f,right:m,type:d===0&&p===0&&f===0&&m===0?"VALID":"EXPLICIT"},u=Or((t-r+d+p)/s+1,a),h=Or((e-i+f+m)/o+1,a)}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:c,outHeight:u,outWidth:h}}function mI(n,t,e,s,o,r,i,a,l,c,u){let h,d,p,f;if(n==="valid"&&(n=0),typeof n=="number"){h={top:n,bottom:n,left:n,right:n,front:n,back:n,type:n===0?"VALID":"NUMBER"};const g=pI([t,e,s,1],[a,l,c],1,[o,r,i],n,u);d=g[0],p=g[1],f=g[2]}else if(n==="same"){d=Math.ceil(t/o),p=Math.ceil(e/r),f=Math.ceil(s/i);const m=(d-1)*o+a-t,g=(p-1)*r+l-e,x=(f-1)*i+c-s,b=Math.floor(m/2),w=m-b,y=Math.floor(g/2),$=g-y,v=Math.floor(x/2),N=x-v;h={top:y,bottom:$,left:v,right:N,front:b,back:w,type:"SAME"}}else throw Error(`Unknown padding parameter: ${n}`);return{padInfo:h,outDepth:d,outHeight:p,outWidth:f}}function Or(n,t){if(!t)return Math.trunc(n);switch(t){case"round":return Math.round(n);case"ceil":return Math.ceil(n);case"floor":return Math.floor(n);default:throw new Error(`Unknown roundingMode ${t}`)}}function co(n){const[t,e,s]=_r(n);return t===1&&e===1&&s===1}function Te(n,t){return co(n)||co(t)}function uo(n){return _r(n).every(t=>t>0)}function us(n){if(n==="NHWC")return"channelsLast";if(n==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${n}`)}function Ge(n,t,e){if(e!=null){if(typeof t=="string")throw Error(`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${t}.`);if(typeof t=="number")I(oo(t),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(o=>{I(oo(o),()=>`Error in ${n}: pad must be an integer when using dimRoundingMode ${e} but got pad ${o}.`)})});else throw Error(`Error in ${n}: Unknown padding parameter: ${t}`)}}function gI(n,t){const s={x:E(n,"x","reshape","string_or_numeric")},o={shape:t};return R.runKernel(sc,s,o)}const L=_({reshape_:gI});function xI(n,t,e,s,o){const r=E(n,"x","avgPool","float32"),i=1;I(Te(e,i),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${e} and dilations '${i}'`);let a=r,l=!1;r.rank===3&&(l=!0,a=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),I(a.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),Ge("avgPool",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o};let h=R.runKernel(yl,c,u);return h=st(h,r.dtype),l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const id=_({avgPool_:xI});function bI(n,t,e,s,o,r="NDHWC"){const i=E(n,"x","avgPool3d","float32");let a=i,l=!1;i.rank===4&&(l=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),I(a.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${a.rank}.`),I(r==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${r}`),I(typeof e=="number"&&e>0||Array.isArray(e)&&e[0]>0&&e[1]>0&&e[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${e}'`),Ge("avgPool3d",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o,dataFormat:r};let h=R.runKernel(wl,c,u);return h=st(h,a.dtype),l?L(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const yI=_({avgPool3d_:bI});function wI(n,t=0){I(n.length>=1,()=>"Pass at least one tensor to concat");const e=nd(n,"tensors","concat","string_or_numeric");if(e[0].dtype==="complex64"&&e.forEach(r=>{if(r.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${r.dtype}. `)}),e.length===1)return eo(e[0]);const s=e,o={axis:t};return R.runKernel(vl,s,o)}const ke=_({concat_:wI});function $I(n,t,e=!1,s=!1){let o=E(n,"a","matMul"),r=E(t,"b","matMul");[o,r]=Qt(o,r);const i={a:o,b:r},a={transposeA:e,transposeB:s};return R.runKernel($l,i,a)}const Et=_({matMul_:$I});function CI(n){const e={x:E(n,"x","sigmoid","float32")};return R.runKernel(Oi,e)}const er=_({sigmoid_:CI});function II(n,t,e){const s=E(n,"x","slice","string_or_numeric");if(s.rank===0)throw new Error("Slicing scalar is not possible");const o={x:s},r={begin:t,size:e};return R.runKernel(lc,o,r)}const _t=_({slice_:II});function vI(n){const e={x:E(n,"x","tanh","float32")};return R.runKernel(Vi,e)}const wc=_({tanh_:vI});function kI(n,t,e){const s=E(n,"x","batchToSpaceND"),o=t.reduce((a,l)=>a*l);I(s.rank>=1+t.length,()=>`input rank is ${s.rank} but should be > than blockShape.length ${t.length}`),I(e.length===t.length,()=>`crops.length is ${e.length} but should be equal to blockShape.length  ${t.length}`),I(s.shape[0]%o===0,()=>`input tensor batch is ${s.shape[0]} but is not divisible by the product of the elements of blockShape ${t.join(" * ")} === ${o}`);const r={x:s},i={blockShape:t,crops:e};return R.runKernel(Cl,r,i)}const ad=_({batchToSpaceND_:kI});function SI(n){let t;return n.rank===0||n.rank===1?t=L(n,[1,1,1,n.size]):n.rank===2?t=L(n,[1,1,n.shape[0],n.shape[1]]):n.rank===3?t=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]):t=n,t}function NI(n,t,e,s,o,r){r==null&&(r=.001);const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;s!=null&&(u=E(s,"offset","batchNorm")),I(a.rank===l.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),I(u==null||a.rank===u.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),I(c==null||a.rank===c.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d={x:SI(i),scale:c,offset:u,mean:a,variance:l},p={varianceEpsilon:r},f=R.runKernel(Fl,d,p);return L(f,i.shape)}const $c=_({batchNorm_:NI});function TI(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),I(i.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${i.rank}.`),I(a.rank===2||a.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${a.rank}.`),I(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${l.rank}.`),c!=null&&I(c.rank===2||c.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${c.rank}.`),u!=null&&I(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${u.rank}.`),$c(i,a,l,u,c,r)}const EI=_({batchNorm2d_:TI});function RI(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),I(i.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${i.rank}.`),I(a.rank===3||a.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${a.rank}.`),I(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${l.rank}.`),c!=null&&I(c.rank===3||c.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${c.rank}.`),u!=null&&I(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${u.rank}.`),$c(i,a,l,u,c,r)}const AI=_({batchNorm3d_:RI});function DI(n,t,e,s,o,r){const i=E(n,"x","batchNorm"),a=E(t,"mean","batchNorm"),l=E(e,"variance","batchNorm");let c;o!=null&&(c=E(o,"scale","batchNorm"));let u;return s!=null&&(u=E(s,"offset","batchNorm")),I(i.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${i.rank}.`),I(a.rank===4||a.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${a.rank}.`),I(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${l.rank}.`),c!=null&&I(c.rank===4||c.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${c.rank}.`),u!=null&&I(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${u.rank}.`),$c(i,a,l,u,c,r)}const FI=_({batchNorm4d_:DI});function _I(n,t,e){const s=E(n,"x","bincount"),o=E(t,"weights","bincount");I(s.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${s.dtype}`),I(e>=0,()=>`size must be non-negative, but got ${e}.`),I(o.size===s.size||o.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${s.shape}, weights shape: ${o.shape}.`);const r={x:s,weights:o},i={size:e};return R.runKernel(lh,r,i)}const OI=_({bincount_:_I});function LI(n,t){let e=E(n,"broadcastTo","x");const s=e.shape;if(Fn(t),t.length<e.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${e.rank}.`);if(t.length>e.rank){const c=e.shape.slice();for(;c.length<t.length;)c.unshift(1);e=L(e,c)}const o=e.shape,r=Array.from(t);for(let c=t.length-1;c>=0;c--)if(o[c]===t[c])r[c]=1;else if(e.shape[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(r.map((c,u)=>c>1?u:-1).filter(c=>c>=0).length===0)return eo(e);const a={x:e},l={reps:r};return R.runKernel(Wi,a,l)}const Er=_({broadcastTo_:LI});function MI(n){const e={x:E(n,"x","ceil","float32")};return R.runKernel(si,e)}const PI=_({ceil_:MI});function qi(n,t,e){Fn(n),e=e||vo(t);const s={shape:n,value:t,dtype:e};return R.runKernel(Ih,{},s)}function zI(n,t,e){const s=E(n,"x","clipByValue");if(I(t<=e,()=>`Error in clip: min (${t}) must be less than or equal to max (${e}).`),t===e)return qi(s.shape,t,s.dtype);const o={x:s},r={clipValueMin:t,clipValueMax:e};return R.runKernel(oi,o,r)}const on=_({clipByValue_:zI});function BI(n){return ke(n,0)}const VI=_({concat1d_:BI});function WI(n,t){return ke(n,t)}const UI=_({concat2d_:WI});function GI(n,t){return ke(n,t)}const HI=_({concat3d_:GI});function qI(n,t){return ke(n,t)}const jI=_({concat4d_:qI});function KI(n,t,e,s,o="NHWC",r=[1,1],i){const a=E(n,"x","conv2d","float32"),l=E(t,"filter","conv2d","float32");let c=a,u=!1;a.rank===3&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),I(c.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${c.rank}.`),I(l.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${l.rank}.`),Ge("conv2d",s,i);const h=o==="NHWC"?c.shape[3]:c.shape[1];I(h===l.shape[2],()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${l.shape[2]}.`),I(Te(e,r),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`),I(uo(r),()=>"Error in conv2D: Dilated rates should be larger than 0."),I(uo(e),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:c,filter:l},p={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i},f=R.runKernel(kl,d,p);return u?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const ho=_({conv2d_:KI});function XI(n,t,e,s,o="NWC",r=1,i){const a=E(n,"x","conv1d"),l=E(t,"filter","conv1d");let c=a,u=!1;a.rank===2&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1]])),I(c.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${c.rank}.`),I(l.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),Ge("conv1d",s,i),I(c.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${c.shape[2]}) must match input depth for filter ${l.shape[1]}.`),I(Te(e,r),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${e} and dilation '${r}'`),I(uo(r),()=>"Error in conv1D: Dilated rates should be larger than 0."),I(uo(e),()=>"Error in conv1D: Stride should be larger than 0."),I(o==="NWC",()=>`Error in conv1d: got dataFormat of ${o} but only NWC is currently supported.`);const h=L(l,[1,l.shape[0],l.shape[1],l.shape[2]]),d=L(c,[c.shape[0],1,c.shape[1],c.shape[2]]),g=ho(d,h,[1,e],s,"NHWC",[1,r],i);return u?L(g,[g.shape[2],g.shape[3]]):L(g,[g.shape[0],g.shape[2],g.shape[3]])}const Pg=_({conv1d_:XI});function YI(n,t,e,s,o,r="NHWC",i){I(n.length===t.rank,()=>`Length of inShape (${n.length}) and rank of dy (${t.rank}) must match`);let a=n,l=t,c=!1;t.rank===3&&(c=!0,l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]),a=[1,n[0],n[1],n[2]]),I(a.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${a.length}.`),I(l.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${l.rank}`),I(e.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${e.rank}`);const u=r==="NHWC"?a[3]:a[1],h=r==="NHWC"?l.shape[3]:l.shape[1];I(u===e.shape[2],()=>`Error in conv2dDerInput: depth of input (${u}) must match input depth for filter ${e.shape[2]}.`),I(h===e.shape[3],()=>`Error in conv2dDerInput: depth of output (${h}) must match output depth for filter ${e.shape[3]}.`),Ge("conv2dDerInput",o,i);const d={dy:l,filter:e},p={strides:s,pad:o,dataFormat:r,dimRoundingMode:i,inputShape:a},f=R.runKernel(Sl,d,p);return c?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const ld=_({conv2DBackpropInput_:YI});function ZI(n,t,e,s,o,r){const i=E(n,"x","conv2dTranspose"),a=E(t,"filter","conv2dTranspose");return ld(e,i,a,s,o,"NHWC",r)}const zg=_({conv2dTranspose_:ZI});function JI(n,t,e,s,o="NDHWC",r=[1,1,1]){const i=E(n,"x","conv3d"),a=E(t,"filter","conv3d");let l=i,c=!1;i.rank===4&&(c=!0,l=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),I(l.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${l.rank}.`),I(a.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${a.rank}.`),I(l.shape[4]===a.shape[3],()=>`Error in conv3d: depth of input (${l.shape[4]}) must match input depth for filter ${a.shape[3]}.`),I(Te(e,r),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`),I(o==="NDHWC",()=>`Error in conv3d: got dataFormat of ${o} but only NDHWC is currently supported.`),I(uo(r),()=>"Error in conv3D: Dilated rates should be larger than 0."),I(uo(e),()=>"Error in conv3D: Strides should be larger than 0.");const u={x:l,filter:a},h={strides:e,pad:s,dataFormat:o,dilations:r},d=R.runKernel(Nl,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const QI=_({conv3d_:JI});function tv(n,t,e,s,o){I(n.length===t.rank,()=>`Length of inShape (${n.length}) and rank of dy (${t.rank}) must match`);let r=n,i=t,a=!1;t.rank===4&&(a=!0,i=L(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]]),r=[1,n[0],n[1],n[2],n[3]]);const l=r[4],c=i.shape[4];I(r.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${r.length}.`),I(i.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${i.rank}`),I(e.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${e.rank}`),I(l===e.shape[3],()=>`Error in conv3dDerInput: depth of input (${l}) must match input depth for filter ${e.shape[3]}.`),I(c===e.shape[4],()=>`Error in conv3dDerInput: depth of output (${c}) must match output depth for filter ${e.shape[4]}.`);const u={dy:i,filter:e},h={pad:o,strides:s,inputShape:r},d=R.runKernel(ph,u,h);return a?L(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Bg=_({conv3DBackpropInput_:tv});function ev(n,t,e,s,o){const r=E(n,"x","conv3dTranspose"),i=E(t,"filter","conv3dTranspose");return Bg(e,r,i,s,o)}const nv=_({conv3dTranspose_:ev});function sv(n){const e={x:E(n,"x","cos","float32")};return R.runKernel(ri,e)}const cd=_({cos_:sv});function ov(n){const e={x:E(n,"x","cosh","float32")};return R.runKernel(ii,e)}const Vg=_({cosh_:ov});function rv(n,t=0,e=!1,s=!1){const r={x:E(n,"x","cumprod")},i={axis:t,exclusive:e,reverse:s};return R.runKernel(fh,r,i)}const Tu=_({cumprod_:rv});function iv(n,t=0,e=!1,s=!1){const r={x:E(n,"x","cumsum")},i={axis:t,exclusive:e,reverse:s};return R.runKernel(Tl,r,i)}const Wg=_({cumsum_:iv});function av(n,t,e,s=!1){const o=E(n,"x","denseBincount"),r=E(t,"weights","denseBincount");I(o.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${o.dtype}`),I(o.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${o.rank}.`),I(e>=0,()=>`size must be non-negative, but got ${e}.`),I(r.size===o.size||r.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${o.shape}, weights shape: ${r.shape}.`);const i={x:o,weights:r},a={size:e,binaryOutput:s};return R.runKernel(gh,i,a)}const wf=_({denseBincount_:av});function lv(n,t,e="NHWC"){const s=E(n,"x","depthToSpace","float32"),o=e==="NHWC"?s.shape[1]:s.shape[2],r=e==="NHWC"?s.shape[2]:s.shape[3],i=e==="NHWC"?s.shape[3]:s.shape[1];I(t>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${t}`),I(o*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${o} and ${t}  for depthToSpace with input shape
    ${s.shape}`),I(r*t>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${t} for depthToSpace with input shape
        ${s.shape}`),I(i%(t*t)===0,()=>`Dimension size must be evenly divisible by ${t*t} but is ${i} for depthToSpace with input shape ${s.shape}`);const a={x:s},l={blockSize:t,dataFormat:e};return R.runKernel(xh,a,l)}const cv=_({depthToSpace_:lv});function uv(n,t,e,s,o="NHWC",r=[1,1],i){const a=E(n,"x","depthwiseConv2d","float32"),l=E(t,"filter","depthwiseConv2d","float32");let c=a,u=!1;a.rank===3&&(u=!0,c=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),I(c.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${c.rank}.`),I(l.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${l.rank}.`);const h=o==="NHWC"?c.shape[3]:c.shape[1];I(h===l.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${h}) must match the inChannels dimension in filter ${l.shape[2]}.`),Ge("depthwiseConv2d",s,i);const d={x:c,filter:l},p={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i},f=R.runKernel(El,d,p);return u?L(f,[f.shape[1],f.shape[2],f.shape[3]]):f}const ud=_({depthwiseConv2d_:uv});function hv(n,t,e,s,o=[1,1],r="NHWC"){const i=E(n,"x","dilation2d"),a=E(t,"filter","dilation2d");I(i.rank===3||i.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${i.rank}.`),I(a.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${a.rank}.`),I(r==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${r}`);let l=i,c=!1;i.rank===3&&(l=L(i,[1,i.shape[0],i.shape[1],i.shape[2]]),c=!0),I(l.shape[3]===a.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${l.shape[3]} vs ${a.shape[2]}`);const u={x:l,filter:a},h={strides:e,pad:s,dilations:o},d=R.runKernel(Rl,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const dv=_({dilation2d_:hv});function po(n,t){const e=n.length,s=[];for(let o=0;o<e;o++){const r=e-1-o,i=n[r]||1;(t[t.length-1-o]||1)>1&&i===1&&s.unshift(r)}return s}function re(n,t){const e=[];for(let s=0;s<t.length;s++){const o=n[n.length-s-1],r=t.length-s-1,i=t[r];(o==null||o===1&&i>1)&&e.unshift(r)}return e}function gt(n,t){const e=Math.max(n.length,t.length),s=new Array(e);for(let o=0;o<e;o++){let r=n[n.length-o-1];r==null&&(r=1);let i=t[t.length-o-1];if(i==null&&(i=1),r===1)s[e-o-1]=i;else if(i===1)s[e-o-1]=r;else if(r!==i){const a=`Operands could not be broadcast together with shapes ${n} and ${t}.`;throw Error(a)}else s[e-o-1]=r}return s}const HK=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:gt,getBroadcastDims:po,getReductionAxes:re},Symbol.toStringTag,{value:"Module"}));function pv(n,t){let e=E(n,"a","equal","string_or_numeric"),s=E(t,"b","equal","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Al,o)}const fn=_({equal_:pv});function fv(n,t,e){const s=E(t,"a","where"),o=E(e,"b","where"),r=E(n,"condition","where","bool"),i=gt(gt(r.shape,s.shape),o.shape),a=Er(r,i),l=Er(s,i),c=Er(o,i),u={condition:a,t:l,e:c};return R.runKernel(ac,u)}const Se=_({where_:fv});function mv(n){const e={x:E(n,"x","zerosLike")};return R.runKernel(gc,e)}const kt=_({zerosLike_:mv});function gv(n,t){let e=E(n,"a","div"),s=E(t,"b","div");[e,s]=Qt(e,s);const o=ct(e,s),r=kt(o),i=fn(s,r);return Se(i,r,o)}const xv=_({divNoNan_:gv});function bv(n,t){const e=E(n,"t1","dot"),s=E(t,"t2","dot");I((e.rank===1||e.rank===2)&&(s.rank===1||s.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${e.rank} and ${s.rank}.`);const o=e.rank===1?e.size:e.shape[1],r=s.rank===1?s.size:s.shape[0];if(I(o===r,()=>`Error in dot: inner dimensions of inputs must match, but got ${o} and ${r}.`),e.rank===1&&s.rank===1){const i=L(e,[1,-1]),a=L(s,[-1,1]),l=Et(i,a);return L(l,[])}else if(e.rank===1&&s.rank===2){const i=L(e,[1,-1]),a=L(s,[s.shape[0],s.shape[1]]),l=Et(i,a);return L(l,[l.size])}else if(e.rank===2&&s.rank===1){const i=L(s,[-1,1]),a=Et(e,i);return L(a,[a.size])}else{const i=L(s,[s.shape[0],s.shape[1]]);return Et(e,i)}}const yv=_({dot_:bv});function wv(n,...t){const e=t.map((o,r)=>E(o,`tensors${r}`,"einsum")),s={equation:n};return R.runKernel(wh,e,s)}const xr=_({einsum_:wv});function $v(n){const e={x:E(n,"x","elu","float32")};return R.runKernel(li,e)}const Cc=_({elu_:$v});function Cv(n){let t=E(n,"x","erf");I(t.dtype==="int32"||t.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),t.dtype==="int32"&&(t=st(t,"float32"));const e={x:t};return R.runKernel(ci,e)}const Ug=_({erf_:Cv});function hd(n,t){for(let e=0;e<n.length;++e)if(n[n.length-e-1]!==t-1-e)return!1;return!0}function Gg(n,t,e){const s=n.length+t.length,o=[];let r=0,i=0;for(let a=0;a<s;a++)e.indexOf(a)===-1?o.push(n[r++]):o.push(t[i++]);return o}function me(n,t){const e=[],s=n.length;for(let r=0;r<s;r++)t.indexOf(r)===-1&&e.push(n[r]);const o=t.map(r=>n[r]);return[e,o]}function ee(n,t){const e=t.map(s=>1);return Gg(n,e,t)}function Ie(n,t,e){I(hd(t,e),()=>`${n} supports only inner-most axes for now. Got axes ${t} and rank-${e} input.`)}function Xt(n,t){if(hd(n,t))return null;const e=[];for(let s=0;s<t;++s)n.indexOf(s)===-1&&e.push(s);return n.forEach(s=>e.push(s)),e}function Ps(n){return n.map((t,e)=>[e,t]).sort((t,e)=>t[1]-e[1]).map(t=>t[0])}function ne(n,t){const e=[];for(let s=t-n;s<t;++s)e.push(s);return e}function Iv(n,t=null,e=!1){const o={x:E(n,"x","max")},r={reductionIndices:t,keepDims:e};return R.runKernel(Ul,o,r)}const Nn=_({max_:Iv});function vv(n,t=null,e=!1){const o={x:E(n,"x","min")},r={axis:t,keepDims:e};return R.runKernel(jl,o,r)}const Ha=_({min_:vv});function kv(n,t){let e=E(n,"base","pow"),s=E(t,"exp","pow");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(ki,o)}const fo=_({pow_:kv});function mt(n,t){if((je(n)&&t!=="string"||Array.isArray(n))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&je(n)&&!(n instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return yc(n,[],[],t)}function Sv(n){const e={x:E(n,"x","sqrt","float32")};return R.runKernel(Mi,e)}const Ne=_({sqrt_:Sv});function Nv(n){const t=E(n,"x","square"),e={};return R.runKernel("Square",{x:t},e)}const Vt=_({square_:Nv});function Tv(n,t=null,e=!1){let s=E(n,"x","sum");s.dtype==="bool"&&(s=st(s,"int32"));const o={x:s},r={axis:t,keepDims:e};return R.runKernel(cc,o,r)}const it=_({sum_:Tv});function Ev(n,t="euclidean",e=null,s=!1){n=E(n,"x","norm");const o=Hg(n,t,e);let r=o.shape;if(s){const i=$t(e,n.shape);r=ee(o.shape,i)}return L(o,r)}function Hg(n,t,e=null){if(n.rank===0)return he(n);if(n.rank!==1&&e===null)return Hg(L(n,[-1]),t,e);if(n.rank===1||typeof e=="number"||Array.isArray(e)&&e.length===1){if(t===1)return it(he(n),e);if(t===1/0)return Nn(he(n),e);if(t===-1/0)return Ha(he(n),e);if(t==="euclidean"||t===2)return Ne(it(fo(he(n),mt(2,"int32")),e));throw new Error(`Error in norm: invalid ord value: ${t}`)}if(Array.isArray(e)&&e.length===2){if(t===1)return Nn(it(he(n),e[0]),e[1]-1);if(t===1/0)return Nn(it(he(n),e[1]),e[0]);if(t===-1/0)return Ha(it(he(n),e[1]),e[0]);if(t==="fro"||t==="euclidean")return Ne(it(Vt(n),e));throw new Error(`Error in norm: invalid ord value: ${t}`)}throw new Error(`Error in norm: invalid axis: ${e}`)}const Ic=_({norm_:Ev});function Rv(n,t=null,e=!1){return Ic(n,"euclidean",t,e)}const Av=_({euclideanNorm_:Rv});function Dv(n){const e={x:E(n,"x","exp")};return R.runKernel(ui,e)}const rn=_({exp_:Dv});function Fv(n,t=0){const e=E(n,"x","expandDims","string_or_numeric");I(t<=e.rank,()=>"Axis must be <= rank of the tensor");const s={input:e},o={dim:t};return R.runKernel(Dl,s,o)}const Me=_({expandDims_:Fv});function _v(n){const e={x:E(n,"x","expm1")};return R.runKernel(hi,e)}const Ov=_({expm1_:_v});function Lv(n,t){const e=E(n,"x","tile","string_or_numeric");I(e.rank===t.length,()=>`Error in transpose: rank of input ${e.rank} must match length of reps ${t}.`);const s={x:e},o={reps:t};return R.runKernel(Wi,s,o)}const kn=_({tile_:Lv});function Mv(n,t,e,s="float32"){t==null&&(t=n);const o=Ct([n,t],s),r=n<=t?n:t;for(let a=0;a<r;++a)o.set(1,a,a);const i=L(o.toTensor(),[n,t]);if(e==null)return i;if(e.length===1)return kn(Me(i,0),[e[0],1,1]);if(e.length===2)return kn(Me(Me(i,0),0),[e[0],e[1],1,1]);if(e.length===3)return kn(Me(Me(Me(i,0),0),0),[e[0],e[1],e[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${e.length}D.`)}const qg=_({eye_:Mv});function Pv(n){const e={x:E(n,"x","floor","float32")};return R.runKernel(di,e)}const vc=_({floor_:Pv});function zv(n,t,e=0,s=0){const o=E(n,"x","gather"),r=E(t,"indices","gather","int32"),i={x:o,indices:r},a={axis:e,batchDims:s};return R.runKernel(_l,i,a)}const dd=_({gather_:zv});function Bv(n,t){let e=E(n,"a","greater","string_or_numeric"),s=E(t,"b","greater","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Ol,o)}const Ze=_({greater_:Bv});function Vv(n,t){let e=E(n,"a","greaterEqual","string_or_numeric"),s=E(t,"b","greaterEqual","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(fi,o)}const So=_({greaterEqual_:Vv});function Wv(n){const e={input:E(n,"input","imag")};return R.runKernel(Nh,e)}const pd=_({imag_:Wv});function Uv(n){const e={x:E(n,"x","isFinite")};return R.runKernel(gi,e)}const Gv=_({isFinite_:Uv});function Hv(n){const e={x:E(n,"x","isInf")};return R.runKernel(xi,e)}const qv=_({isInf_:Hv});function jv(n){const e={x:E(n,"x","isNaN")};return R.runKernel(bi,e)}const Kv=_({isNaN_:jv});function Xv(n,t=.2){const s={x:E(n,"x","leakyRelu")},o={alpha:t};return R.runKernel(Ll,s,o)}const fd=_({leakyRelu_:Xv});function Yv(n,t){let e=E(n,"a","less","string_or_numeric"),s=E(t,"b","less","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Ml,o)}const qa=_({less_:Yv});function Zv(n,t){let e=E(n,"a","lessEqual","string_or_numeric"),s=E(t,"b","lessEqual","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Pl,o)}const nr=_({lessEqual_:Zv});function Jv(n,t=5,e=1,s=1,o=.5){const r=E(n,"x","localResponseNormalization");I(r.rank===4||r.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${r.rank}.`),I(oo(t),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${t}.`);let i=r,a=!1;r.rank===3&&(a=!0,i=L(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const l={x:i},c={depthRadius:t,bias:e,alpha:s,beta:o},u=R.runKernel(Wl,l,c);return a?L(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Qv=_({localResponseNormalization_:Jv});function tk(n){const e={x:E(n,"x","log","float32")};return R.runKernel(yi,e)}const mn=_({log_:tk});function ek(n){const e={x:E(n,"x","log1p")};return R.runKernel(wi,e)}const md=_({log1p_:ek});function qK(n){return I(Ts(n),()=>"The f passed in grad(f) must be a function"),(t,e)=>{const s=E(t,"x","tf.grad","string_or_numeric"),o=e!=null?E(e,"dy","tf.grad"):null;return R.tidy(()=>{const{value:r,grads:i}=R.gradients(()=>n(s),[s],o);return o!=null&&Oe(r.shape,o.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),kc(i),i[0]})}}function jK(n){return I(Ts(n),()=>"The f passed in grads(f) must be a function"),(t,e)=>{I(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=nd(t,"args","tf.grads","string_or_numeric"),o=e!=null?E(e,"dy","tf.grads"):null;return R.tidy(()=>{const{value:r,grads:i}=R.gradients(()=>n(...s),s,o);return o!=null&&Oe(r.shape,o.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),kc(i),i})}}function KK(n){return I(Ts(n),()=>"The f passed in valueAndGrad(f) must be a function"),(t,e)=>{I(t instanceof Lt,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),I(e==null||e instanceof Lt,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:o}=R.gradients(()=>n(t),[t],e);return kc(s),{grad:s[0],value:o}}}function XK(n){return I(Ts(n),()=>"The f passed in valueAndGrads(f) must be a function"),(t,e)=>{I(Array.isArray(t)&&t.every(o=>o instanceof Lt),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),I(e==null||e instanceof Lt,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=R.gradients(()=>n(...t),t,e);return e!=null&&Oe(s.value.shape,e.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),kc(s.grads),s}}function nk(n,t){I(Ts(n),()=>"The f passed in variableGrads(f) must be a function"),I(t==null||Array.isArray(t)&&t.every(c=>c instanceof Ga),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const e=t!=null;if(!e){t=[];for(const c in R.registeredVariables)t.push(R.registeredVariables[c])}const s=e?t.filter(c=>!c.trainable):null,o=t.length;t=t.filter(c=>c.trainable),I(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${o} variables is trainable.`);const r=!0,{value:i,grads:a}=R.gradients(n,t,null,r);I(a.some(c=>c!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),I(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const l={};return t.forEach((c,u)=>{a[u]!=null&&(l[c.name]=a[u])}),s?.forEach(c=>l[c.name]=null),{value:i,grads:l}}function mo(n){return R.customGrad(n)}function kc(n){if(n.filter(e=>e==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}function sk(n){const e={x:E(n,"x","neg")};return R.runKernel(Xl,e)}const Ht=_({neg_:sk});function ok(n){const e={x:E(n,"x","softplus")};return R.runKernel(Li,e)}const ji=_({softplus_:ok});function rk(n){const t=E(n,"x","logSigmoid");return mo(s=>({value:Ht(ji(Ht(s))),gradFunc:i=>A(i,er(Ht(s)))}))(t)}const ik=_({logSigmoid_:rk});function ak(n,t){let e=E(n,"a","sub"),s=E(t,"b","sub");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(zi,o)}const rt=_({sub_:ak});function lk(n,t=-1){const e=E(n,"logits","logSoftmax");if(t===-1&&(t=e.rank-1),t!==e.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${e.rank} and axis was ${t}`);return mo((o,r)=>{const a=Nn(o,t,!0),l=rt(o,a),c=rt(st(l,"float32"),mn(it(rn(l),t,!0)));return r([c]),{value:c,gradFunc:(h,d)=>{const[p]=d,f=!0,m=rn(p);return rt(h,A(it(h,t,f),m))}}})(e)}const jg=_({logSoftmax_:lk});function ck(n,t=null,e=!1){const s=E(n,"x","logSumExp"),o=$t(t,s.shape),r=Nn(s,o,!0),i=rt(s,r),a=rn(i),l=it(a,o),c=mn(l),u=J(L(r,c.shape),c);if(e){const h=ee(u.shape,o);return L(u,h)}return u}const gd=_({logSumExp_:ck});function uk(n,t){const e=E(n,"a","logicalAnd","bool"),s=E(t,"b","logicalAnd","bool");gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(zl,o)}const Gn=_({logicalAnd_:uk});function hk(n){const e={x:E(n,"x","logicalNot","bool")};return R.runKernel(Bl,e)}const xd=_({logicalNot_:hk});function dk(n,t){const e=E(n,"a","logicalOr","bool"),s=E(t,"b","logicalOr","bool");gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Vl,o)}const Kg=_({logicalOr_:dk});function pk(n,t){const e=E(n,"a","logicalXor","bool"),s=E(t,"b","logicalXor","bool");return gt(e.shape,s.shape),Gn(Kg(n,t),xd(Gn(n,t)))}const fk=_({logicalXor_:pk});function mk(n,t,e,s,o){const r=E(n,"x","maxPool"),i=1;let a=r,l=!1;r.rank===3&&(l=!0,a=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),I(a.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),I(Te(e,i),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${e} and dilations '${i}'`),Ge("maxPool",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o},h=R.runKernel(Gl,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const bd=_({maxPool_:mk});function gk(n,t=[1,1,1],e,s,o,r="NDHWC"){const i=E(n,"x","maxPool3d");let a=i,l=!1;i.rank===4&&(l=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),I(a.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${a.rank}.`),I(r==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${r}`),Ge("maxPool3d",s,o);const c={x:a},u={filterSize:t,strides:e,pad:s,dimRoundingMode:o,dataFormat:r},h=R.runKernel(Hl,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const xk=_({maxPool3d_:gk});function bk(n,t){let e=E(n,"a","maximum"),s=E(t,"b","maximum");[e,s]=Qt(e,s),e.dtype==="bool"&&(e=st(e,"int32"),s=st(s,"int32")),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel($i,o)}const zs=_({maximum_:bk});function yk(n,t=null,e=!1){const o={x:E(n,"x","mean")},r={axis:t,keepDims:e};return R.runKernel(ql,o,r)}const te=_({mean_:yk});function pe(n,t="float32"){if(Fn(n),t==="complex64"){const s=pe(n,"float32"),o=pe(n,"float32");return ro(s,o)}const e=$e(U(n),t);return R.makeTensor(e,n,t)}function hs(n,t="float32"){if(Fn(n),t==="complex64"){const s=hs(n,"float32"),o=pe(n,"float32");return ro(s,o)}const e=ml(U(n),t);return R.makeTensor(e,n,t)}function wk(n,t){let e=E(n,"a","minimum"),s=E(t,"b","minimum");[e,s]=Qt(e,s),e.dtype==="bool"&&(e=st(e,"int32"),s=st(s,"int32")),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Ci,o)}const jo=_({minimum_:wk});function $k(n,t,e){I(e==="reflect"||e==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${e}.`);const s=E(n,"x","mirrorPad");if(s.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");I(t.length===s.rank,()=>`Padding doesn't match input. Must be ${s.rank}. Got ${t.length}.`);const o=e==="reflect"?1:0;for(let a=0;a<s.rank;a++)I(t[a].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),I(t[a][0]>=0&&t[a][0]<=s.shape[a]-o&&t[a][1]>=0&&t[a][1]<=s.shape[a]-o,()=>`Padding in dimension ${a} cannot be greater than or equal to ${s.shape[a]-o} or less than 0 for input of shape ${s.shape}`);const r={paddings:t,mode:e},i={x:s};return R.runKernel(Kl,i,r)}const Ck=_({mirrorPad_:$k});function Ik(n,t){let e=E(n,"a","mod"),s=E(t,"b","mod");[e,s]=Qt(e,s);const o={a:e,b:s};return R.runKernel(Ii,o)}const vk=_({mod_:Ik});function kk(n,t=null,e=!1){n=E(n,"x","moments");const s=$t(t,n.shape),o=te(n,s,e);let r=o.shape;e||(r=ee(o.shape,s));const i=Vt(rt(st(n,"float32"),L(o,r))),a=te(i,s,e);return{mean:o,variance:a}}const yd=_({moments_:kk});function Sk(n,t){let e=E(n,"a","notEqual","string_or_numeric"),s=E(t,"b","notEqual","string_or_numeric");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s};return R.runKernel(Yl,o)}const Lr=_({notEqual_:Sk});function Nk(n,t,e=1,s=0,o="int32"){if(t<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${t}`);const i={indices:E(n,"indices","oneHot","int32")},a={dtype:o,depth:t,onValue:e,offValue:s};return R.runKernel(Jl,i,a)}const Xg=_({oneHot_:Nk});function Tk(n){const e={x:E(n,"x","onesLike")};return R.runKernel(Zl,e)}const gn=_({onesLike_:Tk});function Ek(n,t,e=0){const s=E(n,"x","pad");if(s.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const o={paddings:t,constantValue:e},r={x:s};return R.runKernel(tc,r,o)}const wd=_({pad_:Ek});function Rk(n,t,e){const s=E(n,"x","spaceToBatchND");I(s.rank>=1+t.length,()=>`input rank ${s.rank} should be > than [blockShape] ${t.length}`),I(e.length===t.length,()=>`paddings.shape[0] ${e.length} must be equal to [blockShape] ${t.length}`),I(s.shape.reduce((i,a,l)=>l>0&&l<=t.length?i&&(a+e[l-1][0]+e[l-1][1])%t[l-1]===0:i,!0),()=>`input spatial dimensions ${s.shape.slice(1)} with paddings ${e.toString()} must be divisible by blockShapes ${t.toString()}`);const o={x:s},r={blockShape:t,paddings:e};return R.runKernel(uc,o,r)}const $d=_({spaceToBatchND_:Rk});function Ak(n,t,e,s,o,r,i){o==null&&(o=[1,1]),r==null&&(r=1),s===0&&(s="valid");const a=E(n,"x","maxPool");let l=a,c=!1;a.rank===3&&(c=!0,l=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),I(Te(r,o),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${r} and dilations '${o}'`);const u=xn(l.shape,t,r,o,s),h=[u.dilationHeight,u.dilationWidth];let d;s==="same"?d=Fk([u.filterHeight,u.filterWidth],h):d=[[0,0],[0,0]];const p=h[0]===1&&h[1]===1,[f,m]=Dk([u.inHeight,u.inWidth],h,d),g=p?s:"valid",x=p?l:$d(l,h,f),w=(e==="avg"?()=>id(x,t,r,g,i):()=>bd(x,t,r,g,i))(),y=p?w:ad(w,h,m);return c?L(y,[y.shape[1],y.shape[2],y.shape[3]]):y}function Dk(n,t,e){const s=e.map(u=>u[0]),o=e.map(u=>u[1]),r=n.concat(s,o),i=t.map((u,h)=>(u-r[h]%u)%u),a=o.map((u,h)=>u+i[h]),l=t.map((u,h)=>[s[h],a[h]]),c=t.map((u,h)=>[0,i[h]]);return[l,c]}function Fk(n,t){const s=n.map((i,a)=>i+(i-1)*(t[a]-1)).map(i=>i-1),o=s.map(i=>Math.floor(i/2)),r=s.map((i,a)=>i-o[a]);return s.map((i,a)=>[o[a],r[a]])}const _k=_({pool_:Ak});function Ok(n,t){const e=E(n,"x","prelu"),s=E(t,"alpha","prelu"),o={x:e,alpha:s};return R.runKernel(ec,o)}const Cd=_({prelu_:Ok});function Lk(n,t=null,e=!1){let s=E(n,"x","prod");s.dtype==="bool"&&(s=st(s,"int32"));const o={x:s},r={axis:t,keepDims:e};return R.runKernel(nc,o,r)}const Mk=_({prod_:Lk});function Pk(n,t,e){Fn(n);const s=U(n);let o=null;if(e==null||e==="float32")o=new Float32Array(s);else if(e==="int32")o=new Int32Array(s);else if(e==="bool")o=new Uint8Array(s);else throw new Error(`Unknown data type ${e}`);for(let r=0;r<s;r++)o[r]=t();return R.makeTensor(o,n,e)}const YK=_({rand_:Pk});var Sa={exports:{}},zk=Sa.exports,$f;function Bk(){return $f||($f=1,(function(n){(function(t,e,s){function o(l){var c=this,u=a();c.next=function(){var h=2091639*c.s0+c.c*23283064365386963e-26;return c.s0=c.s1,c.s1=c.s2,c.s2=h-(c.c=h|0)},c.c=1,c.s0=u(" "),c.s1=u(" "),c.s2=u(" "),c.s0-=u(l),c.s0<0&&(c.s0+=1),c.s1-=u(l),c.s1<0&&(c.s1+=1),c.s2-=u(l),c.s2<0&&(c.s2+=1),u=null}function r(l,c){return c.c=l.c,c.s0=l.s0,c.s1=l.s1,c.s2=l.s2,c}function i(l,c){var u=new o(l),h=c&&c.state,d=u.next;return d.int32=function(){return u.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,h&&(typeof h=="object"&&r(h,u),d.state=function(){return r(u,{})}),d}function a(){var l=4022871197,c=function(u){u=String(u);for(var h=0;h<u.length;h++){l+=u.charCodeAt(h);var d=.02519603282416938*l;l=d>>>0,d-=l,d*=l,l=d>>>0,d-=l,l+=d*4294967296}return(l>>>0)*23283064365386963e-26};return c}e&&e.exports?e.exports=i:this.alea=i})(zk,n)})(Sa)),Sa.exports}var Na={exports:{}},Vk=Na.exports,Cf;function Wk(){return Cf||(Cf=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.x=0,l.y=0,l.z=0,l.w=0,l.next=function(){var h=l.x^l.x<<11;return l.x=l.y,l.y=l.z,l.z=l.w,l.w^=l.w>>>19^h^h>>>8},a===(a|0)?l.x=a:c+=a;for(var u=0;u<c.length+64;u++)l.x^=c.charCodeAt(u)|0,l.next()}function r(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xor128=i})(Vk,n)})(Na)),Na.exports}var Ta={exports:{}},Uk=Ta.exports,If;function Gk(){return If||(If=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.next=function(){var h=l.x^l.x>>>2;return l.x=l.y,l.y=l.z,l.z=l.w,l.w=l.v,(l.d=l.d+362437|0)+(l.v=l.v^l.v<<4^(h^h<<1))|0},l.x=0,l.y=0,l.z=0,l.w=0,l.v=0,a===(a|0)?l.x=a:c+=a;for(var u=0;u<c.length+64;u++)l.x^=c.charCodeAt(u)|0,u==c.length&&(l.d=l.x<<10^l.x>>>4),l.next()}function r(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l.v=a.v,l.d=a.d,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xorwow=i})(Uk,n)})(Ta)),Ta.exports}var Ea={exports:{}},Hk=Ea.exports,vf;function qk(){return vf||(vf=1,(function(n){(function(t,e,s){function o(a){var l=this;l.next=function(){var u=l.x,h=l.i,d,p;return d=u[h],d^=d>>>7,p=d^d<<24,d=u[h+1&7],p^=d^d>>>10,d=u[h+3&7],p^=d^d>>>3,d=u[h+4&7],p^=d^d<<7,d=u[h+7&7],d=d^d<<13,p^=d^d<<9,u[h]=p,l.i=h+1&7,p};function c(u,h){var d,p=[];if(h===(h|0))p[0]=h;else for(h=""+h,d=0;d<h.length;++d)p[d&7]=p[d&7]<<15^h.charCodeAt(d)+p[d+1&7]<<13;for(;p.length<8;)p.push(0);for(d=0;d<8&&p[d]===0;++d);for(d==8?p[7]=-1:p[d],u.x=p,u.i=0,d=256;d>0;--d)u.next()}c(l,a)}function r(a,l){return l.x=a.x.slice(),l.i=a.i,l}function i(a,l){a==null&&(a=+new Date);var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(u.x&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xorshift7=i})(Hk,n)})(Ea)),Ea.exports}var Ra={exports:{}},jk=Ra.exports,kf;function Kk(){return kf||(kf=1,(function(n){(function(t,e,s){function o(a){var l=this;l.next=function(){var u=l.w,h=l.X,d=l.i,p,f;return l.w=u=u+1640531527|0,f=h[d+34&127],p=h[d=d+1&127],f^=f<<13,p^=p<<17,f^=f>>>15,p^=p>>>12,f=h[d]=f^p,l.i=d,f+(u^u>>>16)|0};function c(u,h){var d,p,f,m,g,x=[],b=128;for(h===(h|0)?(p=h,h=null):(h=h+"\0",p=0,b=Math.max(b,h.length)),f=0,m=-32;m<b;++m)h&&(p^=h.charCodeAt((m+32)%h.length)),m===0&&(g=p),p^=p<<10,p^=p>>>15,p^=p<<4,p^=p>>>13,m>=0&&(g=g+1640531527|0,d=x[m&127]^=p+g,f=d==0?f+1:0);for(f>=128&&(x[(h&&h.length||0)&127]=-1),f=127,m=512;m>0;--m)p=x[f+34&127],d=x[f=f+1&127],p^=p<<13,d^=d<<17,p^=p>>>15,d^=d>>>12,x[f]=p^d;u.w=g,u.X=x,u.i=f}c(l,a)}function r(a,l){return l.i=a.i,l.w=a.w,l.X=a.X.slice(),l}function i(a,l){a==null&&(a=+new Date);var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(u.X&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.xor4096=i})(jk,n)})(Ra)),Ra.exports}var Aa={exports:{}},Xk=Aa.exports,Sf;function Yk(){return Sf||(Sf=1,(function(n){(function(t,e,s){function o(a){var l=this,c="";l.next=function(){var h=l.b,d=l.c,p=l.d,f=l.a;return h=h<<25^h>>>7^d,d=d-p|0,p=p<<24^p>>>8^f,f=f-h|0,l.b=h=h<<20^h>>>12^d,l.c=d=d-p|0,l.d=p<<16^d>>>16^f,l.a=f-h|0},l.a=0,l.b=0,l.c=-1640531527,l.d=1367130551,a===Math.floor(a)?(l.a=a/4294967296|0,l.b=a|0):c+=a;for(var u=0;u<c.length+20;u++)l.b^=c.charCodeAt(u)|0,l.next()}function r(a,l){return l.a=a.a,l.b=a.b,l.c=a.c,l.d=a.d,l}function i(a,l){var c=new o(a),u=l&&l.state,h=function(){return(c.next()>>>0)/4294967296};return h.double=function(){do var d=c.next()>>>11,p=(c.next()>>>0)/4294967296,f=(d+p)/(1<<21);while(f===0);return f},h.int32=c.next,h.quick=h,u&&(typeof u=="object"&&r(u,c),h.state=function(){return r(c,{})}),h}e&&e.exports?e.exports=i:this.tychei=i})(Xk,n)})(Aa)),Aa.exports}var Da={exports:{}};const Zk={},Jk=Object.freeze(Object.defineProperty({__proto__:null,default:Zk},Symbol.toStringTag,{value:"Module"})),Qk=d$(Jk);var tS=Da.exports,Nf;function eS(){return Nf||(Nf=1,(function(n){(function(t,e,s){var o=256,r=6,i=52,a="random",l=s.pow(o,r),c=s.pow(2,i),u=c*2,h=o-1,d;function p(y,$,v){var N=[];$=$==!0?{entropy:!0}:$||{};var T=x(g($.entropy?[y,w(e)]:y??b(),3),N),k=new f(N),S=function(){for(var C=k.g(r),D=l,O=0;C<c;)C=(C+O)*o,D*=o,O=k.g(1);for(;C>=u;)C/=2,D/=2,O>>>=1;return(C+O)/D};return S.int32=function(){return k.g(4)|0},S.quick=function(){return k.g(4)/4294967296},S.double=S,x(w(k.S),e),($.pass||v||function(C,D,O,P){return P&&(P.S&&m(P,k),C.state=function(){return m(k,{})}),O?(s[a]=C,D):C})(S,T,"global"in $?$.global:this==s,$.state)}function f(y){var $,v=y.length,N=this,T=0,k=N.i=N.j=0,S=N.S=[];for(v||(y=[v++]);T<o;)S[T]=T++;for(T=0;T<o;T++)S[T]=S[k=h&k+y[T%v]+($=S[T])],S[k]=$;(N.g=function(C){for(var D,O=0,P=N.i,M=N.j,V=N.S;C--;)D=V[P=h&P+1],O=O*o+V[h&(V[P]=V[M=h&M+D])+(V[M]=D)];return N.i=P,N.j=M,O})(o)}function m(y,$){return $.i=y.i,$.j=y.j,$.S=y.S.slice(),$}function g(y,$){var v=[],N=typeof y,T;if($&&N=="object")for(T in y)try{v.push(g(y[T],$-1))}catch{}return v.length?v:N=="string"?y:y+"\0"}function x(y,$){for(var v=y+"",N,T=0;T<v.length;)$[h&T]=h&(N^=$[h&T]*19)+v.charCodeAt(T++);return w($)}function b(){try{var y;return d&&(y=d.randomBytes)?y=y(o):(y=new Uint8Array(o),(t.crypto||t.msCrypto).getRandomValues(y)),w(y)}catch{var $=t.navigator,v=$&&$.plugins;return[+new Date,t,v,t.screen,w(e)]}}function w(y){return String.fromCharCode.apply(0,y)}if(x(s.random(),e),n.exports){n.exports=p;try{d=Qk}catch{}}else s["seed"+a]=p})(typeof self<"u"?self:tS,[],Math)})(Da)),Da.exports}var su,Tf;function nS(){if(Tf)return su;Tf=1;var n=Bk(),t=Wk(),e=Gk(),s=qk(),o=Kk(),r=Yk(),i=eS();return i.alea=n,i.xor128=t,i.xorwow=e,i.xorshift7=s,i.xor4096=o,i.tychei=r,su=i,su}var sr=nS();class Id{constructor(t,e,s,o,r){this.mean=t,this.stdDev=e,this.dtype=s,this.nextVal=NaN,this.truncated=o,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const i=r||Math.random();this.random=sr.alea(i.toString())}nextValue(){if(!isNaN(this.nextVal)){const o=this.nextVal;return this.nextVal=NaN,o}let t,e,s=!1;for(;!s;){let o,r,i;do o=2*this.random()-1,r=2*this.random()-1,i=o*o+r*r;while(i>=1||i===0);const a=Math.sqrt(-2*Math.log(i)/i);t=this.mean+this.stdDev*o*a,e=this.mean+this.stdDev*r*a,(!this.truncated||this.isValidTruncated(t))&&(s=!0)}return(!this.truncated||this.isValidTruncated(e))&&(this.nextVal=this.convertValue(e)),this.convertValue(t)}convertValue(t){return this.dtype==null||this.dtype==="float32"?t:Math.round(t)}isValidTruncated(t){return t<=this.upper&&t>=this.lower}}class ZK{constructor(t,e,s,o){this.alpha=t,this.beta=1/e,this.dtype=s;const r=o||Math.random();this.randu=sr.alea(r.toString()),this.randn=new Id(0,1,s,!1,this.randu()),t<1?this.d=t+2/3:this.d=t-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let t,e,s,o,r,i;for(;;){do o=this.randn.nextValue(),i=1+this.c*o;while(i<=0);if(i*=i*i,t=o*o,e=1-.331*t*t,s=.5*t+this.d*(1-i+Math.log(i)),r=this.randu(),r<e||Math.log(r)<s)break}return i=1/this.beta*this.d*i,this.alpha<1&&(i*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(i)}convertValue(t){return this.dtype==="float32"?t:Math.round(t)}}class sS{constructor(t=0,e=1,s,o){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=t,this.range=e-t,this.dtype=s,o==null&&(o=Math.random()),typeof o=="number"&&(o=o.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${t} - ${e} <= 1 and dtype is not float`);this.random=sr.alea(o)}convertValue(t){return this.canReturnFloat()?t:Math.round(t)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}function oS(n,t=0,e=1,s,o){if(Fn(n),s!=null&&s==="bool")throw new Error(`Unsupported data type ${s}`);const r=new Id(t,e,s,!1,o),i=Ct(n,s);for(let a=0;a<i.values.length;a++)i.values[a]=r.nextValue();return i.toTensor()}const rS=_({randomNormal_:oS});function iS(n,t=0,e=1,s="float32",o){Fn(n);const r=Ct(n,s),i=new sS(t,e,null,o);for(let a=0;a<r.values.length;a++)r.values[a]=i.nextValue();return r.toTensor()}const Ki=_({randomUniform_:iS});function Mr(n,t,e=1,s="float32"){if(e===0)throw new Error("Cannot have a step of zero");const o={start:n,stop:t,step:e,dtype:s};return R.runKernel(_h,{},o)}function aS(n){const e={input:E(n,"input","real")};return R.runKernel(Oh,e)}const ja=_({real_:aS});function lS(n){const e={x:E(n,"x","reciprocal")};return R.runKernel(Si,e)}const cS=_({reciprocal_:lS});function uS(n){const e={x:E(n,"x","relu")};return R.runKernel(Ni,e)}const ds=_({relu_:uS});function hS(n){const e={x:E(n,"x","relu6")};return R.runKernel(Ti,e)}const Yg=_({relu6_:hS});function dS(n,t){const s={x:E(n,"x","reverse")},o={dims:t};return R.runKernel(ic,s,o)}const go=_({reverse_:dS});function pS(n){const e={x:E(n,"x","round")};return R.runKernel(Ei,e)}const Zg=_({round_:pS});function fS(n){const e={x:E(n,"x","rsqrt","float32")};return R.runKernel(Ri,e)}const Jg=_({rsqrt_:fS});function mS(n){const e={x:E(n,"x","selu")};return R.runKernel(Ai,e)}const Qg=_({selu_:mS});function gS(n,t,e,s,o,r=[1,1],i="NHWC"){const a=E(n,"x","separableConv2d"),l=E(t,"depthwiseFilter","separableConv2d"),c=E(e,"pointwiseFilter","separableConv2d");let u=a,h=!1;if(a.rank===3&&(h=!0,u=L(a,[1,a.shape[0],a.shape[1],a.shape[2]])),i==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");I(u.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${u.rank}.`),I(l.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),I(c.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),I(c.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${c.shape[0]}.`),I(c.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${c.shape[1]}.`);const d=l.shape[2],p=l.shape[3];I(c.shape[2]===d*p,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*p}, but got ${c.shape[2]}.`);const f=ud(u,l,s,o,i,r),g=ho(f,c,1,"valid",i);return h?L(g,[g.shape[1],g.shape[2],g.shape[3]]):g}const tx=_({separableConv2d_:gS});function xS(n){const e={x:E(n,"x","sign")};return R.runKernel(_i,e)}const bS=_({sign_:xS});function yS(n){const e={x:E(n,"x","sin","float32")};return R.runKernel(Di,e)}const ex=_({sin_:yS});function wS(n){const e={x:E(n,"x","sinh")};return R.runKernel(Fi,e)}const nx=_({sinh_:wS});function $S(n,t,e){const s=E(n,"x","slice1d");return I(s.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${s.rank} tensor`),_t(s,[t],[e])}const vd=_({slice1d_:$S});function CS(n,t,e){const s=E(n,"x","slice2d");return I(s.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${s.rank} tensor`),_t(s,t,e)}const sx=_({slice2d_:CS});function IS(n,t,e){const s=E(n,"x","slice3d");return I(s.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${s.rank} tensor`),_t(s,t,e)}const kd=_({slice3d_:IS});function vS(n,t,e){const s=E(n,"x","slice4d");return I(s.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${s.rank} tensor`),_t(s,t,e)}const Ka=_({slice4d_:vS});function kS(n,t=-1){const e=E(n,"logits","softmax","float32");if(t===-1&&(t=e.rank-1),t!==e.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${e.rank} and dim was ${t}`);const s={logits:e},o={dim:t};return R.runKernel(dc,s,o)}const Sd=_({softmax_:kS});function SS(n){I(n.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${n.dtype}.`);const t={input:n};return R.runKernel(Ch,t)}const Nd=_({fft_:SS});function NS(n){I(n.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${n.dtype}.`);const t={input:n};return R.runKernel(Sh,t)}const Xa=_({ifft_:NS});function TS(n){const t=n.shape[n.shape.length-1],e=n.size/t;let s;if(t<=2){const o=L(n,[e,t]);s=Xa(o)}else{const o=[e,2*(t-1)],r=L(ja(n),[e,t]),i=L(pd(n),[e,t]),a=go(_t(r,[0,1],[e,t-2]),1),l=A(go(_t(i,[0,1],[e,t-2]),1),mt(-1)),c=ke([r,a],1),u=ke([i,l],1),h=L(ro(c,u),[o[0],o[1]]);s=Xa(h)}if(s=ja(s),n.rank===3&&n.shape[0]!==0){const o=s,r=n.shape[0];s=L(s,[r,s.shape[0]/r,s.shape[1]]),o.dispose()}return s}const ox=_({irfft_:TS});function ES(n,t,e=0){const o={x:E(n,"x","split")},r={numOrSizeSplits:t,axis:e};return R.runKernel(hc,o,r)}const sn=_({split_:ES});function RS(n,t){I(n.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${n.dtype}`);let e=n.shape[n.shape.length-1];const s=n.size/e;let o;if(t!=null&&t<e){const f=n.shape.map(g=>0),m=n.shape.map(g=>g);m[n.shape.length-1]=t,o=_t(n,f,m),e=t}else if(t!=null&&t>e){const f=n.shape.map(m=>m);f[n.shape.length-1]=t-e,o=ke([n,pe(f)],n.shape.length-1),e=t}else o=n;const r=kt(o),i=L(ro(o,r),[s,e]),a=Nd(i),l=Math.floor(e/2)+1,c=ja(a),u=pd(a),h=sn(c,[l,e-l],c.shape.length-1),d=sn(u,[l,e-l],u.shape.length-1),p=o.shape.slice();return p[o.shape.length-1]=l,L(ro(h[0],d[0]),p)}const Td=_({rfft_:RS});function AS(n,t){let e=E(n,"a","squaredDifference"),s=E(t,"b","squaredDifference");[e,s]=Qt(e,s),gt(e.shape,s.shape);const o={a:e,b:s},r={};return R.runKernel(Pi,o,r)}const rx=_({squaredDifference_:AS});function DS(n,t){const e=E(n,"x","squeeze","string_or_numeric");return L(e,is(e.shape,t).newShape)}const Xi=_({squeeze_:DS});function FS(n,t=0){const e=nd(n,"tensors","stack","string_or_numeric");I(e.length>=1,()=>"Pass at least one tensor to tf.stack"),e.length>0&&I(t<=e[0].rank,()=>"Axis must be <= rank of the tensor");const s=e,o={axis:t};return R.runKernel(Ql,s,o)}const Hn=_({stack_:FS});function _S(n,t=0){const s={x:E(n,"x","step")},o={alpha:t};return R.runKernel(Ui,s,o)}const Yi=_({step_:_S});function OS(n,t,e,s,o=0,r=0,i=0,a=0,l=0){const u={x:E(n,"x","stridedSlice","string_or_numeric")},h={begin:t,end:e,strides:s,beginMask:o,endMask:r,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:l};return R.runKernel(Uh,u,h)}const LS=_({stridedSlice_:OS});function MS(n){const e={x:E(n,"x","tan","float32")};return R.runKernel(Bi,e)}const PS=_({tan_:MS});function Be(n,t){Ju(n);const e=bc(n,t);if(e.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return yc(n,null,e,t)}function Fa(n,t,e){if(Ju(n),t!=null&&t.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const s=bc(n,e);if(s.length!==2&&s.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(s.length===1&&t==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return yc(n,t,s,e)}function Ed(n,t,e){const s=t.rank>1?t.shape[t.rank-1]:1,o=t.rank>1?t.rank-1:1,r=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${e.shape}, indices.shape: ${t.shape}, shape: ${n}, sliceDim: ${s}, and batchDim: ${o}.`;if(e.rank<o)throw new Error(r+` update.rank < ${o}. `);if(n.length<s+(e.rank-o))throw new Error(r+` Output shape length < ${s+(e.rank-o)}`);if(e.rank!==o+n.length-s)throw new Error(r+` update.rank != ${o+n.length-s}`);for(let i=0;i<o;++i)if(e.shape[i]!==t.shape[i])throw new Error(r+` updates.shape[${i}] (${e.shape[i]}) != indices.shape[${i}] (${t.shape[i]}).`);for(let i=0;i<e.rank-o;++i)if(e.shape[i+o]!==n[i+s])throw new Error(r+` updates.shape[${i+o}] (${e.shape[i+o]}) != shape[${i+o}] (${n[i+o]})`)}function ix(n,t,e){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(n.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${n.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(e.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${e}`);if(e.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(n.size===0)throw new Error(`Updates specified for empty output. updates shape: ${n.shape}`)}Ed(e,t,n)}function Bs(n,t,e){const s=t.shape.length,o=s>1?t.shape[s-1]:1,r=e.length;let i=1;for(let h=o;h<r;++h)i*=e[h];const a=o<1?1:o,l=U(t.shape)/a,c=[...ut(e.slice(0,o)),1],u=U(e);return{sliceRank:o,numUpdates:l,sliceSize:i,strides:c,outputSize:u}}const JK=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:Bs,validateInput:ix,validateUpdateShape:Ed},Symbol.toStringTag,{value:"Module"}));function zS(n,t=1,e=!0){const s=E(n,"x","topk");if(s.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const o=s.shape[s.shape.length-1];if(t<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${t}`);if(t>o)throw new Error(`'k' passed to topk() must be <= the last dimension (${o}) but got ${t}`);const r={x:s},i={k:t,sorted:e},[a,l]=R.runKernel(jh,r,i);return{values:a,indices:l}}const BS=_({topk_:zS});function VS(n,t=0,e=1,s,o){if(Fn(n),s!=null&&s==="bool")throw new Error("Unsupported data type $ { dtype }");const r=new Id(t,e,s,!0,o),i=Ct(n,s);for(let a=0;a<i.values.length;a++)i.values[a]=r.nextValue();return i.toTensor()}const ax=_({truncatedNormal_:VS});function WS(n,t=0){const e=E(n,"x","unique","string_or_numeric");I(e.rank>0,()=>"The input tensor must be at least 1D");const s={x:e},o={axis:t},[r,i]=R.runKernel(Xh,s,o);return{values:r,indices:i}}const US=_({unique_:WS});function GS(n,t,e){const s=E(n,"x","unsortedSegmentSum"),o=E(t,"segmentIds","unsortedSegmentSum","int32");I(oo(e),()=>"numSegments must be of dtype int");const r={x:s,segmentIds:o},i={numSegments:e};return R.runKernel(mc,r,i)}const lx=_({unsortedSegmentSum_:GS});function HS(n,t=0){const e=E(n,"x","unstack","string_or_numeric");I(t>=-e.shape.length&&t<e.shape.length,()=>`Axis = ${t} is not in [-${e.shape.length}, ${e.shape.length})`);const s={value:e},o={axis:t};return R.runKernel(fc,s,o)}const xo=_({unstack_:HS});function qS(n,t=!0,e,s){return R.makeVariable(n,t,e,s)}function cx(n,t){const e=[];for(let r=0;r<t.length;r++)t[r]&&e.push(r);const s=Ct(n,"int32"),o=Ct([e.length,n.length],"int32");for(let r=0;r<e.length;r++){const i=s.indexToLoc(e[r]),a=r*n.length;o.values.set(i,a)}return o.toTensor()}function jS(n,t,e){const s=E(n,"x","transpose");if(t==null&&(t=s.shape.map((i,a)=>a).reverse()),I(s.rank===t.length,()=>`Error in transpose: rank of input ${s.rank} must match length of perm ${t}.`),t.forEach(i=>{I(i>=0&&i<s.rank,()=>`All entries in 'perm' must be between 0 and ${s.rank-1} but got ${t}`)}),s.rank<=1)return s.clone();const o={x:s},r={perm:t};return s.dtype==="complex64"?z(()=>{let i=ja(s),a=pd(s);return i=R.runKernel(zo,{x:i},r),a=R.runKernel(zo,{x:a},r),e&&(a=Ht(a)),ro(i,a)}):R.runKernel(zo,o,r)}const vt=_({transpose_:jS});function KS(n,t){const e=E(t,"indices","gatherND","int32"),o={params:E(n,"x","gatherND","string_or_numeric"),indices:e};return R.runKernel(kh,o)}const QK=_({gatherND_:KS});function XS(n,t){if(t==null)return n.shape.slice();if(Tt(n.shape,t))return t;if(n.shape.length===t.length){const e=[];for(let s=0;s<n.shape.length;s++)t[s]==null&&n.shape[s]!=null?e.push(n.shape[s]):e.push(t[s]);return e}return t}function YS(n,t,e,s){const o=E(n,"x","dropout");if(I(o.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${o.dtype} tensor instead.`),I(t>=0&&t<1,()=>`rate must be a float in the range [0, 1), but got ${t}.`),t===0)return n instanceof Lt?o.clone():o;const r=XS(o,e),i=1-t,a=ct(vc(J(Ki(r,0,1,"float32",s),i)),i);return A(o,a)}const ZS=_({dropout_:YS});function JS(n){return Math.floor(Math.pow(2,Math.ceil(Math.log(n)/Math.log(2))))}function ux(n,t,e){const s=1-n%2,o=new Float32Array(n);for(let r=0;r<n;++r){const i=2*Math.PI*r/(n+s-1);o[r]=t-e*Math.cos(i)}return Be(o,"float32")}function QS(n,t,e,s,o,r="NHWC",i){let a=n;n.rank===3&&(a=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]));let l=t;l.rank===3&&(l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]])),I(a.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${a.shape}.`),I(l.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${l.shape}.`),I(e.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${e}.`);const c=r==="NHWC"?a.shape[3]:a.shape[1],u=r==="NHWC"?l.shape[3]:l.shape[1];I(c===e[2],()=>`Error in conv2dDerFilter: depth of input ${c}) must match input depth in filter (${e[2]}.`),I(u===e[3],()=>`Error in conv2dDerFilter: depth of dy (${u}) must match output depth for filter (${e[3]}).`),Ge("conv2dDerFilter",o,i);const h={x:a,dy:l},d={strides:s,pad:o,dataFormat:r,dimRoundingMode:i,filterShape:e};return R.runKernel(hh,h,d)}const Rd=_({conv2DBackpropFilter_:QS});function Ad(n,t,e){if(e==null||e==="linear")return n;if(e==="relu")return A(n,Yi(t));throw new Error(`Cannot compute gradient for fused activation ${e}.`)}function Dd(n,t){let e=t;const s=re(n.shape,t.shape);return s.length>0&&(e=it(e,s)),L(e,n.shape)}function Fd(n,t,e,s){if(t==="linear")return n;if(t==="relu")return ds(n);if(t==="elu")return Cc(n);if(t==="relu6")return Yg(n);if(t==="prelu")return Cd(n,e);if(t==="leakyrelu")return fd(n,s);if(t==="sigmoid")return er(n);throw new Error(`Unknown fused activation ${t}.`)}const _d=(n,t)=>!(n>0)||t==="linear";function t2({x:n,filter:t,strides:e,pad:s,dataFormat:o="NHWC",dilations:r=[1,1],dimRoundingMode:i,bias:a,activation:l="linear",preluActivationWeights:c,leakyreluAlpha:u}){if(l=l||"linear",_d(R.state.gradientDepth,l)===!1){I(o==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${o} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let v=ho(n,t,e,s,o,r,i);return a!=null&&(v=J(v,a)),Fd(v,l,c,u)}const h=E(n,"x","conv2d","float32"),d=E(t,"filter","conv2d","float32");let p=h,f=!1;h.rank===3&&(f=!0,p=L(h,[1,h.shape[0],h.shape[1],h.shape[2]])),I(p.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${p.rank}.`),I(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),Ge("fused conv2d",s,i);const m=o==="NHWC"?p.shape[3]:p.shape[1];I(d.shape[2]===m,()=>`Error in conv2d: depth of input (${m}) must match input depth for filter ${d.shape[2]}.`),I(Te(e,r),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${e} and dilations '${r}'`);const g=Ce(p.shape,d.shape,e,r,s,i);let x;a!=null&&(x=E(a,"bias","fused conv2d"),[x]=Qt(x,h),o==="NHWC"?gt(g.outShape,x.shape):(I(x.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${x.shape.length}.`),I(x.shape.length===0||x.shape[0]===g.outChannels||x.shape[0]===1,()=>`Error in fused conv2d: bias shape (${x.shape}) is not compatible with the number of output channels (${g.outChannels})`)));let b;if(c!=null){const v=c.shape;if(I(v.length<=1||v.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${v.length}.`),v.length===1)I(v[0]===1||v[0]===g.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${v}) is not compatible with the number of output channels (${g.outChannels}).`);else if(v.length===3)try{gt(v,g.outShape)}catch{const T=`Error in fused conv2d: PReLU activation weights (${v}) is not compatible with the output shape of the conv2d (${g.outShape}).`;throw Error(T)}b=E(c,"prelu weights","fused conv2d")}const w=(v,N)=>{I(o==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${o} but only NHWC is currently supported.`);const[T,k,S,C]=N,D=Ad(v,S,l);I(co(r),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`);const O=ld(k.shape,D,T,e,s),P=Rd(k,D,T.shape,e,s),M=[O,P];if(C!=null){const V=Dd(C,D);M.push(V)}return M},y={x:p,filter:d,bias:x,preluActivationWeights:b},$={strides:e,pad:s,dataFormat:o,dilations:r,dimRoundingMode:i,activation:l,leakyreluAlpha:u};return a==null?mo((N,T,k)=>{let S=R.runKernel(Ua,y,$);return k([T,N,S]),f&&(S=L(S,[S.shape[1],S.shape[2],S.shape[3]])),{value:S,gradFunc:w}})(p,d):mo((N,T,k,S)=>{let C=R.runKernel(Ua,y,$);return S([T,N,C,k]),f&&(C=L(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:w}})(p,d,x)}const e2=_({fusedConv2d_:t2});function n2(n,t,e,s,o,r=[1,1],i){let a=n;n.rank===3&&(a=L(n,[1,n.shape[0],n.shape[1],n.shape[2]]));let l=t;l.rank===3&&(l=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={x:a,dy:l},u={strides:s,pad:o,dimRoundingMode:i,dilations:r,filterShape:e};return R.runKernel(bh,c,u)}const s2=_({depthwiseConv2dNativeBackpropFilter_:n2});function o2(n,t,e,s,o,r=[1,1],i){let a=t,l=!1;t.rank===3&&(l=!0,a=L(t,[1,t.shape[0],t.shape[1],t.shape[2]]));const c={dy:a,filter:e},u={strides:s,pad:o,dimRoundingMode:i,dilations:r,inputShape:n},h=R.runKernel(yh,c,u);return l?L(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const r2=_({depthwiseConv2dNativeBackpropInput_:o2});function i2({a:n,b:t,transposeA:e=!1,transposeB:s=!1,bias:o,activation:r="linear",preluActivationWeights:i,leakyreluAlpha:a=.2}){if(_d(R.state.gradientDepth,r)===!1){let C=Et(n,t,e,s);return o!=null&&(C=J(C,o)),Fd(C,r,i,a)}let l=E(n,"a","fused matMul"),c=E(t,"b","fused matMul");[l,c]=Qt(l,c);const u=e?l.shape[l.rank-2]:l.shape[l.rank-1],h=s?c.shape[c.rank-1]:c.shape[c.rank-2],d=e?l.shape[l.rank-1]:l.shape[l.rank-2],p=s?c.shape[c.rank-2]:c.shape[c.rank-1],f=l.shape.slice(0,-2),m=c.shape.slice(0,-2),g=U(f),x=U(m);I(u===h,()=>`Error in fused matMul: inner shapes (${u}) and (${h}) of Tensors with shapes ${l.shape} and ${c.shape} and transposeA=${e} and transposeB=${s} must match.`);const w=gt(l.shape.slice(0,-2),c.shape.slice(0,-2)).concat([d,p]),y=e?L(l,[g,u,d]):L(l,[g,d,u]),$=s?L(c,[x,p,h]):L(c,[x,h,p]);let v;o!=null&&(v=E(o,"bias","fused matMul"),[v]=Qt(v,l),gt(w,v.shape));let N;i!=null&&(N=E(i,"prelu weights","fused matMul"));const T=(C,D)=>{const[O,P,M,V]=D,H=Ad(L(C,M.shape),M,r);let W,q;if(!e&&!s?(W=Et(H,P,!1,!0),q=Et(O,H,!0,!1)):!e&&s?(W=Et(H,P,!1,!1),q=Et(H,O,!0,!1)):e&&!s?(W=Et(P,H,!1,!0),q=Et(O,H,!1,!1)):(W=Et(P,H,!0,!0),q=Et(H,O,!0,!0)),o!=null){const j=Dd(V,H);return[W,q,j]}else return[W,q]},k={a:y,b:$,bias:v,preluActivationWeights:N},S={transposeA:e,transposeB:s,activation:r,leakyreluAlpha:a};return o==null?mo((D,O,P)=>{const M=R.runKernel(Wa,k,S);return P([D,O,M]),{value:L(M,w),gradFunc:T}})(y,$):mo((D,O,P,M)=>{const V=R.runKernel(Wa,k,S);return M([D,O,V,P]),{value:L(V,w),gradFunc:T}})(y,$,v)}const Ef=_({fusedMatMul_:i2});function a2(n){return ux(n,.54,.46)}const l2=_({hammingWindow_:a2});function c2(n){return ux(n,.5,.5)}const hx=_({hannWindow_:c2});function u2(n,t,e,s=!1,o=0){let r=0;const i=[];for(;r+t<=n.size;)i.push(_t(n,r,t)),r+=e;if(s)for(;r<n.size;){const a=r+t-n.size,l=ke([_t(n,r,t-a),qi([a],o)]);i.push(l),r+=e}return i.length===0?Fa([],[0,t]):L(ke(i),[i.length,t])}const dx=_({frame_:u2});function h2(n,t,e,s,o=hx){s==null&&(s=JS(t));const r=dx(n,t,e),i=A(r,o(t));return Td(i,s)}const d2=_({stft_:h2});function p2(n,t,e,s,o="bilinear",r=0){const i=E(n,"image","cropAndResize"),a=E(t,"boxes","cropAndResize","float32"),l=E(e,"boxInd","cropAndResize","int32"),c=a.shape[0];I(i.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${i.rank}.`),I(a.rank===2&&a.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${c},4] but had shape ${a.shape}.`),I(l.rank===1&&l.shape[0]===c,()=>`Error in cropAndResize: boxInd must be have size [${c}] but had shape ${a.shape}.`),I(s.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${s.length}.`),I(s[0]>=1&&s[1]>=1,()=>`cropSize must be atleast [1,1], but was ${s}`),I(o==="bilinear"||o==="nearest",()=>`method must be bilinear or nearest, but was ${o}`);const u={image:i,boxes:a,boxInd:l},h={method:o,extrapolationValue:r,cropSize:s};return R.runKernel(mh,u,h)}const f2=_({cropAndResize_:p2});function m2(n){const t=E(n,"image","flipLeftRight","float32");I(t.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${t.rank}.`);const e={image:t};return R.runKernel(vh,e,{})}const g2=_({flipLeftRight_:m2});function x2(n){const t=E(n,"image","grayscaleToRGB"),e=t.rank-1,s=t.shape[e];I(t.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${t.rank}.`),I(s===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${s}.`);const o=new Array(t.rank);return o.fill(1,0,e),o[e]=3,kn(t,o)}const b2=_({grayscaleToRGB_:x2});function y2(n){const t=E(n,"image","RGBToGrayscale"),e=t.rank-1,s=t.shape[e];I(t.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${t.rank}.`),I(s===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${s}.`);const o=t.dtype,r=st(t,"float32"),i=Be([.2989,.587,.114]);let a;switch(t.rank){case 2:a=xr("ij,j->i",r,i);break;case 3:a=xr("ijk,k->ij",r,i);break;case 4:a=xr("ijkl,l->ijk",r,i);break;case 5:a=xr("ijklm,m->ijkl",r,i);break;case 6:a=xr("ijklmn,n->ijklm",r,i);break;default:throw new Error("Not a valid tensor rank.")}return a=Me(a,-1),st(a,o)}const w2=_({rgbToGrayscale_:y2});function $2(n,t,e=0,s=.5){const o=E(n,"image","rotateWithOffset","float32");I(o.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${o.rank}.`);const r={image:o},i={radians:t,fillValue:e,center:s};return R.runKernel(Yh,r,i)}const C2=_({rotateWithOffset_:$2});function or(n,t,e,s,o,r){s==null&&(s=.5),o==null&&(o=Number.NEGATIVE_INFINITY),r==null&&(r=0);const i=n.shape[0];return e=Math.min(e,i),I(0<=s&&s<=1,()=>`iouThreshold must be in [0, 1], but was '${s}'`),I(n.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${n.rank}'`),I(n.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${n.shape[1]}`),I(t.rank===1,()=>"scores must be a 1D tensor"),I(t.shape[0]===i,()=>`scores has incompatible shape with boxes. Expected ${i}, but was ${t.shape[0]}`),I(0<=r&&r<=1,()=>`softNmsSigma must be in [0, 1], but was '${r}'`),{maxOutputSize:e,iouThreshold:s,scoreThreshold:o,softNmsSigma:r}}function I2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY){const r=E(n,"boxes","nonMaxSuppression","float32"),i=E(t,"scores","nonMaxSuppression","float32"),a=or(r,i,e,s,o);e=a.maxOutputSize,s=a.iouThreshold,o=a.scoreThreshold;const l={maxOutputSize:e,iouThreshold:s,scoreThreshold:o};return R.runKernel(Ah,{boxes:r,scores:i},l)}const v2=_({nonMaxSuppression_:I2});function k2(n,t,e){const s=S2(n,t,e),o=s<0?-(s+1):s;n.splice(o,0,t)}function S2(n,t,e){return T2(n,t,e||N2)}function N2(n,t){return n>t?1:n<t?-1:0}function T2(n,t,e){let s=0,o=n.length,r=0,i=!1;for(;s<o;){r=s+(o-s>>>1);const a=e(t,n[r]);a>0?s=r+1:(o=r,i=!a)}return i?s:-s-1}function Od(n,t,e,s,o){return Pd(n,t,e,s,o,0)}function Ld(n,t,e,s,o,r){return Pd(n,t,e,s,o,0,!1,r,!0)}function Md(n,t,e,s,o,r){return Pd(n,t,e,s,o,r,!0)}function Pd(n,t,e,s,o,r,i=!1,a=!1,l=!1){const c=[];for(let g=0;g<t.length;g++)t[g]>o&&c.push({score:t[g],boxIndex:g,suppressBeginIndex:0});c.sort(Rf);const u=r>0?-.5/r:0,h=[],d=[];for(;h.length<e&&c.length>0;){const g=c.pop(),{score:x,boxIndex:b,suppressBeginIndex:w}=g;if(x<o)break;let y=!1;for(let $=h.length-1;$>=w;--$){const v=E2(n,b,h[$]);if(v>=s){y=!0;break}if(g.score=g.score*R2(s,u,v),g.score<=o)break}g.suppressBeginIndex=h.length,y||(g.score===x?(h.push(b),d.push(g.score)):g.score>o&&k2(c,g,Rf))}const p=h.length,f=e-p;a&&f>0&&(h.push(...new Array(f).fill(0)),d.push(...new Array(f).fill(0)));const m={selectedIndices:h};return i&&(m.selectedScores=d),l&&(m.validOutputs=p),m}function E2(n,t,e){const s=n.subarray(t*4,t*4+4),o=n.subarray(e*4,e*4+4),r=Math.min(s[0],s[2]),i=Math.min(s[1],s[3]),a=Math.max(s[0],s[2]),l=Math.max(s[1],s[3]),c=Math.min(o[0],o[2]),u=Math.min(o[1],o[3]),h=Math.max(o[0],o[2]),d=Math.max(o[1],o[3]),p=(a-r)*(l-i),f=(h-c)*(d-u);if(p<=0||f<=0)return 0;const m=Math.max(r,c),g=Math.max(i,u),x=Math.min(a,h),b=Math.min(l,d),w=Math.max(x-m,0)*Math.max(b-g,0);return w/(p+f-w)}function R2(n,t,e){const s=Math.exp(t*e*e);return e<=n?s:0}function Rf(n,t){return n.score-t.score||n.score===t.score&&t.boxIndex-n.boxIndex}async function A2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY){const r=E(n,"boxes","nonMaxSuppressionAsync"),i=E(t,"scores","nonMaxSuppressionAsync"),a=or(r,i,e,s,o);e=a.maxOutputSize,s=a.iouThreshold,o=a.scoreThreshold;const l=await Promise.all([r.data(),i.data()]),c=l[0],u=l[1],{selectedIndices:h}=Od(c,u,e,s,o);return r!==n&&r.dispose(),i!==t&&i.dispose(),Be(h,"int32")}const D2=A2;function F2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=0){const i=E(n,"boxes","nonMaxSuppression"),a=E(t,"scores","nonMaxSuppression"),l=or(i,a,e,s,o,r);e=l.maxOutputSize,s=l.iouThreshold,o=l.scoreThreshold,r=l.softNmsSigma;const c={boxes:i,scores:a},u={maxOutputSize:e,iouThreshold:s,scoreThreshold:o,softNmsSigma:r},h=R.runKernel(Fh,c,u);return{selectedIndices:h[0],selectedScores:h[1]}}const _2=_({nonMaxSuppressionWithScore_:F2});async function O2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=0){const i=E(n,"boxes","nonMaxSuppressionAsync"),a=E(t,"scores","nonMaxSuppressionAsync"),l=or(i,a,e,s,o,r);e=l.maxOutputSize,s=l.iouThreshold,o=l.scoreThreshold,r=l.softNmsSigma;const c=await Promise.all([i.data(),a.data()]),u=c[0],h=c[1],{selectedIndices:d,selectedScores:p}=Md(u,h,e,s,o,r);return i!==n&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:Be(d,"int32"),selectedScores:Be(p)}}const L2=O2;function M2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=!1){const i=E(n,"boxes","nonMaxSuppression"),a=E(t,"scores","nonMaxSuppression"),l=or(i,a,e,s,o,null),c=l.maxOutputSize,u=l.iouThreshold,h=l.scoreThreshold,d={boxes:i,scores:a},p={maxOutputSize:c,iouThreshold:u,scoreThreshold:h,padToMaxOutputSize:r},f=R.runKernel(Dh,d,p);return{selectedIndices:f[0],validOutputs:f[1]}}const P2=_({nonMaxSuppressionPadded_:M2});async function z2(n,t,e,s=.5,o=Number.NEGATIVE_INFINITY,r=!1){const i=E(n,"boxes","nonMaxSuppressionAsync"),a=E(t,"scores","nonMaxSuppressionAsync"),l=or(i,a,e,s,o,null),c=l.maxOutputSize,u=l.iouThreshold,h=l.scoreThreshold,[d,p]=await Promise.all([i.data(),a.data()]),{selectedIndices:f,validOutputs:m}=Ld(d,p,c,u,h,r);return i!==n&&i.dispose(),a!==t&&a.dispose(),{selectedIndices:Be(f,"int32"),validOutputs:mt(m,"int32")}}const B2=z2;function V2(n,t,e=!1,s=!1){const o=E(n,"images","resizeBilinear");I(o.rank===3||o.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${o.rank}.`),I(t.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${t}.`),I(s===!1||e===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let r=o,i=!1;o.rank===3&&(i=!0,r=L(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const a={images:r},l={alignCorners:e,halfPixelCenters:s,size:t},c=R.runKernel(rc,a,l);return i?L(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const px=_({resizeBilinear_:V2});function W2(n,t,e=!1,s=!1){const o=E(n,"images","resizeNearestNeighbor");I(o.rank===3||o.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${o.rank}.`),I(t.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${t}.`),I(o.dtype==="float32"||o.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),I(s===!1||e===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let r=o,i=!1;o.rank===3&&(i=!0,r=L(o,[1,o.shape[0],o.shape[1],o.shape[2]]));const a={images:r},l={alignCorners:e,halfPixelCenters:s,size:t},c=R.runKernel(oc,a,l);return i?L(c,[c.shape[1],c.shape[2],c.shape[3]]):c}const fx=_({resizeNearestNeighbor_:W2});function U2(n,t="binary",e=!1,s=.5){const o=E(n,"image","threshold"),r=.2989,i=.587,a=.114,l=o.shape[0]*o.shape[1];let c=A(Be([s]),255),u,h,d,p;if(I(o.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${o.rank}.`),I(o.shape[2]===3||o.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${o.shape[2]}.`),I(o.dtype==="int32"||o.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${o.dtype}.`),I(t==="otsu"||t==="binary",()=>`Method must be binary or otsu, but was ${t}`),o.shape[2]===3){[u,h,d]=sn(o,[1,1,1],-1);const g=A(u,r),x=A(h,i),b=A(d,a);p=J(J(g,x),b)}else p=n;if(t==="otsu"){const g=OI(st(Zg(p),"int32"),Bo([]),256);c=G2(g,l)}const f=e?nr(p,c):Ze(p,c);return st(A(f,255),"int32")}function G2(n,t){let e=Be([-1]),s=Be([0]),o=Be([0]),r,i,a,l,c,u;for(let h=0;h<n.size-1;h++){r=_t(n,0,h+1),i=_t(n,h+1),c=ct(it(r),t),u=ct(it(i),t);const d=it(A(r,Mr(0,r.size)));a=ct(d,it(r));const p=qi(i.shape,r.size),f=J(Mr(0,i.size),p),m=A(i,f);l=ct(it(m),it(i));const g=rt(a,l),x=rt(a,l),b=A(c,u);o=A(A(b,g),x);const w=Ze(o,s);s=Se(w,o,s),e=Se(w,Be([h]),e)}return e}const H2=_({threshold_:U2});function q2(n,t,e="nearest",s="constant",o=0,r){const i=E(n,"image","transform","float32"),a=E(t,"transforms","transform","float32");I(i.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${i.rank}.`),I(a.rank===2&&(a.shape[0]===i.shape[0]||a.shape[0]===1)&&a.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),I(r==null||r.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${r}.`);const l={image:i,transforms:a},c={interpolation:e,fillMode:s,fillValue:o,outputShape:r};return R.runKernel(Kh,l,c)}const j2=_({transform_:q2});function K2(n,t,e){const s=E(n,"a","bandPart");I(s.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${s.rank}.`);const o=s.shape,[r,i]=s.shape.slice(-2);let a,l;typeof t=="number"?(I(t%1===0,()=>`bandPart(): numLower must be an integer, got ${t}.`),I(t<=r,()=>`bandPart(): numLower (${t}) must not be greater than the number of rows (${r}).`),a=E(t<0?r:t,"numLower","bandPart")):(I(t.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),a=Se(qa(t,0),r,jo(t,r))),typeof e=="number"?(I(e%1===0,()=>`bandPart(): numUpper must be an integer, got ${e}.`),I(e<=i,()=>`bandPart(): numUpper (${e}) must not be greater than the number of columns (${i}).`),l=E(e<0?i:e,"numUpper","bandPart")):(I(e.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),l=Se(qa(e,0),i,jo(e,i)));const c=L(Mr(0,r,1,"int32"),[-1,1]),u=Mr(0,i,1,"int32"),h=rt(c,u),d=Gn(nr(h,a),So(h,Ht(l))),p=pe([r,i],s.dtype);return L(Hn(xo(L(s,[-1,r,i])).map(f=>Se(d,f,p))),o)}const X2=_({bandPart_:K2});function Y2(n){let t;if(Array.isArray(n)){t=!1,I(n!=null&&n.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const o=n[0].shape[0];for(let r=1;r<n.length;++r)I(n[r].shape[0]===o,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${n[r].shape[0]} vs. ${o})`)}else t=!0,n=sn(n,n.shape[0],0).map(o=>Xi(o,[0]));I(n.length<=n[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${n.length}) exceeds number of dimensions (${n[0].shape[0]}).`);const e=[],s=n;for(let o=0;o<n.length;++o)e.push(R.tidy(()=>{let r=s[o];if(o>0)for(let i=0;i<o;++i){const a=A(it(A(e[i],r)),e[i]);r=rt(r,a)}return ct(r,Ic(r,"euclidean"))}));return t?Hn(e,0):e}const Z2=_({gramSchmidt_:Y2});function J2(n,t=!1){if(I(n.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${n.rank}`),n.rank===2)return Af(n,t);{const e=n.shape.slice(0,n.shape.length-2).reduce((l,c)=>l*c),s=xo(L(n,[e,n.shape[n.shape.length-2],n.shape[n.shape.length-1]]),0),o=[],r=[];s.forEach(l=>{const[c,u]=Af(l,t);o.push(c),r.push(u)});const i=L(Hn(o,0),n.shape),a=L(Hn(r,0),n.shape);return[i,a]}}function Af(n,t=!1){return R.tidy(()=>{I(n.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${n.shape.length}D Tensor.`);const e=n.shape[0],s=n.shape[1];let o=qg(e),r=eo(n);const i=Fa([[1]],[1,1]);let a=eo(i);const l=e>=s?s:e;for(let c=0;c<l;++c){const u=r,h=a,d=o;[a,r,o]=R.tidy(()=>{const p=_t(r,[c,c],[e-c,1]),f=Ic(p),m=_t(r,[c,c],[1,1]),g=Se(Ze(m,0),Fa([[-1]]),Fa([[1]])),x=rt(m,A(g,f)),b=ct(p,x);b.shape[0]===1?a=eo(i):a=ke([i,_t(b,[1,0],[b.shape[0]-1,b.shape[1]])],0);const w=Ht(ct(Et(g,x),f)),y=_t(r,[c,0],[e-c,s]),$=A(w,a),v=vt(a);if(c===0)r=rt(y,Et($,Et(v,y)));else{const k=rt(y,Et($,Et(v,y)));r=ke([_t(r,[0,0],[c,s]),k],0)}const N=vt($),T=_t(o,[0,c],[e,o.shape[1]-c]);if(c===0)o=rt(T,Et(Et(T,a),N));else{const k=rt(T,Et(Et(T,a),N));o=ke([_t(o,[0,0],[e,c]),k],1)}return[a,r,o]}),xt([u,h,d])}return!t&&e>s&&(o=_t(o,[0,0],[e,s]),r=_t(r,[0,0],[s,s])),[o,r]})}const Q2=_({qr_:J2});var ze;(function(n){n[n.NONE=0]="NONE",n[n.MEAN=1]="MEAN",n[n.SUM=2]="SUM",n[n.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(ze||(ze={}));function tN(n,t,e=ze.SUM_BY_NONZERO_WEIGHTS){const s=E(n,"losses","computeWeightedLoss");let o=null;t!=null&&(o=E(t,"weights","computeWeightedLoss"));const r=o==null?s:A(s,o);if(e===ze.NONE)return r;if(e===ze.SUM)return it(r);if(e===ze.MEAN){if(o==null)return te(r);{const i=s.size/o.size,a=ct(it(r),it(o));return i>1?ct(a,mt(i)):a}}if(e===ze.SUM_BY_NONZERO_WEIGHTS){if(o==null)return ct(it(r),mt(s.size));{const i=A(o,hs(s.shape)),a=st(it(Lr(i,mt(0))),"float32");return ct(it(r),a)}}throw Error(`Unknown reduction: ${e}`)}const ps=_({computeWeightedLoss_:tN});function eN(n,t,e,s=ze.SUM_BY_NONZERO_WEIGHTS){const o=E(n,"labels","absoluteDifference"),r=E(t,"predictions","absoluteDifference");let i=null;e!=null&&(i=E(e,"weights","absoluteDifference")),Oe(o.shape,r.shape,"Error in absoluteDifference: ");const a=he(rt(o,r));return ps(a,i,s)}const nN=_({absoluteDifference_:eN});function sN(n,t,e,s,o=ze.SUM_BY_NONZERO_WEIGHTS){const r=E(n,"labels","cosineDistance"),i=E(t,"predictions","cosineDistance");let a=null;s!=null&&(a=E(s,"weights","cosineDistance")),Oe(r.shape,i.shape,"Error in cosineDistance: ");const l=mt(1),c=rt(l,it(A(r,i),e,!0));return ps(c,a,o)}const oN=_({cosineDistance_:sN});function rN(n,t,e,s=ze.SUM_BY_NONZERO_WEIGHTS){let o=E(n,"labels","hingeLoss");const r=E(t,"predictions","hingeLoss");let i=null;e!=null&&(i=E(e,"weights","hingeLoss")),Oe(o.shape,r.shape,"Error in hingeLoss: ");const a=mt(1);o=rt(A(mt(2),o),a);const l=ds(rt(a,A(o,r)));return ps(l,i,s)}const iN=_({hingeLoss_:rN});function aN(n,t,e,s=1,o=ze.SUM_BY_NONZERO_WEIGHTS){const r=E(n,"labels","huberLoss"),i=E(t,"predictions","huberLoss");let a=null;e!=null&&(a=E(e,"weights","huberLoss")),Oe(r.shape,i.shape,"Error in huberLoss: ");const l=mt(s),c=he(rt(i,r)),u=jo(c,l),h=rt(c,u),d=J(A(mt(.5),Vt(u)),A(l,h));return ps(d,a,o)}const lN=_({huberLoss_:aN});function cN(n,t,e,s=1e-7,o=ze.SUM_BY_NONZERO_WEIGHTS){const r=E(n,"labels","logLoss"),i=E(t,"predictions","logLoss");let a=null;e!=null&&(a=E(e,"weights","logLoss")),Oe(r.shape,i.shape,"Error in logLoss: ");const l=mt(1),c=mt(s),u=Ht(A(r,mn(J(i,c)))),h=A(rt(l,r),mn(J(rt(l,i),c))),d=rt(u,h);return ps(d,a,o)}const uN=_({logLoss_:cN});function hN(n,t,e,s=ze.SUM_BY_NONZERO_WEIGHTS){const o=E(n,"labels","meanSquaredError"),r=E(t,"predictions","meanSquaredError");let i=null;e!=null&&(i=E(e,"weights","meanSquaredError")),Oe(o.shape,r.shape,"Error in meanSquaredError: ");const a=rx(o,r);return ps(a,i,s)}const dN=_({meanSquaredError_:hN});function pN(n,t){const e=E(n,"labels","sigmoidCrossEntropyWithLogits"),s=E(t,"logits","sigmoidCrossEntropyWithLogits");Oe(e.shape,s.shape,"Error in sigmoidCrossEntropyWithLogits: ");const o=ds(s),r=A(s,e),i=md(rn(Ht(he(s))));return J(rt(o,r),i)}function fN(n,t,e,s=0,o=ze.SUM_BY_NONZERO_WEIGHTS){let r=E(n,"multiClassLabels","sigmoidCrossEntropy");const i=E(t,"logits","sigmoidCrossEntropy");let a=null;if(e!=null&&(a=E(e,"weights","sigmoidCrossEntropy")),Oe(r.shape,i.shape,"Error in sigmoidCrossEntropy: "),s>0){const c=mt(s),u=mt(1),h=mt(.5);r=J(A(r,rt(u,c)),A(h,c))}const l=pN(r,i);return ps(l,a,o)}const mN=_({sigmoidCrossEntropy_:fN});function gN(n,t,e=-1){if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${t.rank} and dim was ${e}`);return mo((o,r,i)=>{const l=gd(r,[e],!0),c=rt(st(r,"float32"),l);i([o,c]);const u=Ht(A(c,o));return{value:it(u,[e]),gradFunc:(p,f)=>{const[m,g]=f,x=ee(p.shape,[e]);return[A(L(p,x),rt(st(m,"float32"),rn(g))),A(L(p,x),rt(rn(g),st(m,"float32")))]}}})(n,t)}function xN(n,t,e,s=0,o=ze.SUM_BY_NONZERO_WEIGHTS){let r=E(n,"onehotLabels","softmaxCrossEntropy");const i=E(t,"logits","softmaxCrossEntropy");let a=null;if(e!=null&&(a=E(e,"weights","softmaxCrossEntropy")),Oe(r.shape,i.shape,"Error in softmaxCrossEntropy: "),s>0){const c=mt(s),u=mt(1),h=mt(r.shape[1]);r=J(A(r,rt(u,c)),ct(c,h))}const l=gN(r,i);return ps(l,a,o)}const bN=_({softmaxCrossEntropy_:xN});function yN(n,t,e,s){const o=E(n,"indices","sparseFillEmptyRows","int32"),r=E(t,"values","sparseFillEmptyRows"),i=E(e,"denseShape","sparseFillEmptyRows","int32"),a=E(s,"defaultValue","sparseFillEmptyRows",r.dtype);if(o.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${o.shape}`);if(r.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${r.shape}`);if(i.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${i.shape}`);if(a.rank!==0)throw new Error(`Default value should be a scalar but received shape ${a.shape}`);const l={indices:o,values:r,denseShape:i,defaultValue:a},c=R.runKernel(Ph,l);return{outputIndices:c[0],outputValues:c[1],emptyRowIndicator:c[2],reverseIndexMap:c[3]}}const wN=_({sparseFillEmptyRows_:yN});function $N(n,t,e){const s=E(n,"inputIndices","sparseReshape","int32"),o=E(t,"inputShape","sparseReshape","int32"),r=E(e,"newShape","sparseReshape","int32");if(s.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${s.shape}`);if(o.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${o.shape}`);if(r.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${r.shape}`);const i={inputIndices:s,inputShape:o,newShape:r},a=R.runKernel(zh,i);return{outputIndices:a[0],outputShape:a[1]}}const CN=_({sparseReshape_:$N});function IN(n,t,e){const s=E(n,"data","sparseSegmentMean"),o=E(t,"indices","sparseSegmentMean","int32"),r=E(e,"segmentIds","sparseSegmentMean","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${o.shape}`);if(r.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${r.shape}`);const i={data:s,indices:o,segmentIds:r};return R.runKernel(Bh,i)}const vN=_({sparseSegmentMean_:IN});function kN(n,t,e){const s=E(n,"data","sparseSegmentSum"),o=E(t,"indices","sparseSegmentSum","int32"),r=E(e,"segmentIds","sparseSegmentSum","int32");if(s.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${o.shape}`);if(r.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${r.shape}`);const i={data:s,indices:o,segmentIds:r};return R.runKernel(Vh,i)}const SN=_({sparseSegmentSum_:kN});function NN(n,t,e,s,o,r,i,a){const l=E(n,"data","stringNGrams","string");if(l.dtype!=="string")throw new Error("Data must be of datatype string");if(l.shape.length!==1)throw new Error(`Data must be a vector, saw: ${l.shape}`);const c=E(t,"dataSplits","stringNGrams");if(c.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const u={separator:e,nGramWidths:s,leftPad:o,rightPad:r,padWidth:i,preserveShortSequences:a},h={data:l,dataSplits:c},d=R.runKernel(Gh,h,u);return{nGrams:d[0],nGramsSplits:d[1]}}const TN=_({stringNGrams_:NN});function EN(n,t,e=!0){const s=E(n,"input","stringSplit","string"),o=E(t,"delimiter","stringSplit","string");if(s.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${s.shape}`);if(o.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${o.shape}`);const r={skipEmpty:e},i={input:s,delimiter:o},a=R.runKernel(Hh,i,r);return{indices:a[0],values:a[1],shape:a[2]}}const RN=_({stringSplit_:EN});function AN(n,t){const e=E(n,"input","stringToHashBucketFast","string"),s={numBuckets:t};if(t<=0)throw new Error("Number of buckets must be at least 1");const o={input:e};return R.runKernel(qh,o,s)}const DN=_({stringToHashBucketFast_:AN});function FN(n,t,e,s=!0){const o=E(n,"input","staticRegexReplace","string"),r={pattern:t,rewrite:e,replaceGlobal:s};return R.runKernel(pc,{x:o},r)}const _N=_({staticRegexReplace_:FN});const t6={fft:Nd,ifft:Xa,rfft:Td,irfft:ox},e6={hammingWindow:l2,hannWindow:hx,frame:dx,stft:d2},ts={flipLeftRight:g2,grayscaleToRGB:b2,resizeNearestNeighbor:fx,resizeBilinear:px,rgbToGrayscale:w2,rotateWithOffset:C2,cropAndResize:f2,nonMaxSuppression:v2,nonMaxSuppressionAsync:D2,nonMaxSuppressionWithScore:_2,nonMaxSuppressionWithScoreAsync:L2,nonMaxSuppressionPadded:P2,nonMaxSuppressionPaddedAsync:B2,threshold:H2,transform:j2},ON={bandPart:X2,gramSchmidt:Z2,qr:Q2},n6={absoluteDifference:nN,computeWeightedLoss:ps,cosineDistance:oN,hingeLoss:iN,huberLoss:lN,logLoss:uN,meanSquaredError:dN,sigmoidCrossEntropy:mN,softmaxCrossEntropy:bN},s6={sparseFillEmptyRows:wN,sparseReshape:CN,sparseSegmentMean:vN,sparseSegmentSum:SN},o6={stringNGrams:TN,stringSplit:RN,stringToHashBucketFast:DN,staticRegexReplace:_N};const LN=new Map,Eu=new Map;class No{getClassName(){return this.constructor.className}static fromConfig(t,e){return new t(e)}}class tn{constructor(){this.classNameMap={}}static getMap(){return tn.instance==null&&(tn.instance=new tn),tn.instance}static register(t){tn.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function X(n,t,e){I(n.className!=null,()=>"Class being registered does not have the static className property defined."),I(typeof n.className=="string",()=>"className is required to be a string, but got type "+typeof n.className),I(n.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof e>"u"&&(e=n.className);const s=e,o=t+">"+s;return tn.register(n),LN.set(o,n),Eu.set(n,o),n}function MN(n){return Eu.has(n)?Eu.get(n):n.className}const r6=Object.freeze(Object.defineProperty({__proto__:null,Serializable:No,SerializationMap:tn,getRegisteredName:MN,registerClass:X},Symbol.toStringTag,{value:"Module"}));class Vs extends No{minimize(t,e=!1,s){const{value:o,grads:r}=this.computeGradients(t,s);if(s!=null){const i=s.map(a=>({name:a.name,tensor:r[a.name]}));this.applyGradients(i)}else this.applyGradients(r);return xt(r),e?o:(o.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,e){return nk(t,e)}dispose(){this.iterations_!=null&&xt(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:mt(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(Vs,Symbol.hasInstance,{value:n=>n.minimize!=null&&n.computeGradients!=null&&n.applyGradients!=null});class mx extends Vs{static get className(){return"Adadelta"}constructor(t,e,s=null){super(),this.learningRate=t,this.rho=e,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=R.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=R.registeredVariables[s],i=!1;this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accum_grad`,variable:z(()=>kt(r).variable(i))}),this.accumulatedUpdates[o]==null&&(this.accumulatedUpdates[o]={originalName:`${s}/accum_var`,variable:z(()=>kt(r).variable(i))});const a=Array.isArray(t)?t[o].tensor:t[s];if(a==null)return;const l=this.accumulatedGrads[o].variable,c=this.accumulatedUpdates[o].variable;z(()=>{const u=J(A(l,this.rho),A(Vt(a),1-this.rho)),h=A(ct(Ne(J(c,this.epsilon)),Ne(J(l,this.epsilon))),a),d=J(A(c,this.rho),A(Vt(h),1-this.rho));l.assign(u),c.assign(d);const p=J(A(h,-this.learningRate),r);r.assign(p)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(xt(this.accumulatedGrads.map(t=>t.variable)),xt(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedUpdates=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,e){return new t(e.learningRate,e.rho,e.epsilon)}}class gx extends Vs{static get className(){return"Adagrad"}constructor(t,e=.1){super(),this.learningRate=t,this.initialAccumulatorValue=e,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=R.registeredVariables[s];this.accumulatedGrads[o]==null&&(this.accumulatedGrads[o]={originalName:`${s}/accumulator`,variable:z(()=>qi(r.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(t)?t[o].tensor:t[s];if(i==null)return;const a=this.accumulatedGrads[o].variable;z(()=>{const l=J(a,Vt(i));a.assign(l);const c=J(A(ct(i,Ne(J(l,R.backend.epsilon()))),-this.learningRate),r);r.assign(c)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&xt(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(e)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,e){return new t(e.learningRate,e.initialAccumulatorValue)}}class xx extends Vs{static get className(){return"Adam"}constructor(t,e,s,o=null){super(),this.learningRate=t,this.beta1=e,this.beta2=s,this.epsilon=o,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],z(()=>{this.accBeta1=mt(e).variable(),this.accBeta2=mt(s).variable()}),o==null&&(this.epsilon=R.backend.epsilon())}applyGradients(t){const e=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);z(()=>{const s=rt(1,this.accBeta1),o=rt(1,this.accBeta2);e.forEach((r,i)=>{const a=R.registeredVariables[r],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${r}/m`,variable:z(()=>kt(a).variable(l))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${r}/v`,variable:z(()=>kt(a).variable(l))});const c=Array.isArray(t)?t[i].tensor:t[r];if(c==null)return;const u=this.accumulatedFirstMoment[i].variable,h=this.accumulatedSecondMoment[i].variable,d=J(A(u,this.beta1),A(c,1-this.beta1)),p=J(A(h,this.beta2),A(Vt(c),1-this.beta2)),f=ct(d,s),m=ct(p,o);u.assign(d),h.assign(p);const g=J(A(ct(f,J(Ne(m),this.epsilon)),-this.learningRate),a);a.assign(g)}),this.accBeta1.assign(A(this.accBeta1,this.beta1)),this.accBeta2.assign(A(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&xt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&xt(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t),z(()=>{this.accBeta1.assign(fo(this.beta1,this.iterations_+1)),this.accBeta2.assign(fo(this.beta2,this.iterations_+1))});const e=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,e){return new t(e.learningRate,e.beta1,e.beta2,e.epsilon)}}class bx extends Vs{static get className(){return"Adamax"}constructor(t,e,s,o=null,r=0){super(),this.learningRate=t,this.beta1=e,this.beta2=s,this.epsilon=o,this.decay=r,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],z(()=>{this.iteration=mt(0).variable(),this.accBeta1=mt(e).variable()}),o==null&&(this.epsilon=R.backend.epsilon())}applyGradients(t){const e=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);z(()=>{const s=rt(1,this.accBeta1),o=ct(-this.learningRate,J(A(this.iteration,this.decay),1));e.forEach((r,i)=>{const a=R.registeredVariables[r],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${r}/m`,variable:kt(a).variable(l)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${r}/v`,variable:kt(a).variable(l)});const c=Array.isArray(t)?t[i].tensor:t[r];if(c==null)return;const u=this.accumulatedFirstMoment[i].variable,h=this.accumulatedWeightedInfNorm[i].variable,d=J(A(u,this.beta1),A(c,1-this.beta1)),p=A(h,this.beta2),f=he(c),m=zs(p,f);u.assign(d),h.assign(m);const g=J(A(ct(o,s),ct(d,J(m,this.epsilon))),a);a.assign(g)}),this.iteration.assign(J(this.iteration,1)),this.accBeta1.assign(A(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&xt(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&xt(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,e){return new t(e.learningRate,e.beta1,e.beta2,e.epsilon,e.decay)}}class zd extends Vs{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=Array.isArray(t)?t[o].tensor:t[s];if(r==null)return;const i=R.registeredVariables[s];z(()=>{const a=J(A(this.c,r),i);i.assign(a)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=Bn(mt(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,e){return new t(e.learningRate)}}class yx extends zd{static get className(){return"Momentum"}constructor(t,e,s=!1){super(t),this.learningRate=t,this.momentum=e,this.useNesterov=s,this.accumulations=[],this.m=mt(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=R.registeredVariables[s];this.accumulations[o]==null&&(this.accumulations[o]={originalName:`${s}/momentum`,variable:z(()=>kt(r).variable(!1))});const i=this.accumulations[o].variable,a=Array.isArray(t)?t[o].tensor:t[s];a!=null&&z(()=>{let l;const c=J(A(this.m,i),a);this.useNesterov?l=J(A(this.c,J(a,A(c,this.m))),r):l=J(A(this.c,c),r),i.assign(c),r.assign(l)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&xt(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(e)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,e){return new t(e.learningRate,e.momentum,e.useNesterov)}}class wx extends Vs{static get className(){return"RMSProp"}constructor(t,e=.9,s=0,o=null,r=!1){if(super(),this.learningRate=t,this.decay=e,this.momentum=s,this.epsilon=o,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=r,o==null&&(this.epsilon=R.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,o)=>{const r=R.registeredVariables[s],i=!1;this.accumulatedMeanSquares[o]==null&&(this.accumulatedMeanSquares[o]={originalName:`${s}/rms`,variable:z(()=>kt(r).variable(i))}),this.accumulatedMoments[o]==null&&(this.accumulatedMoments[o]={originalName:`${s}/momentum`,variable:z(()=>kt(r).variable(i))}),this.accumulatedMeanGrads[o]==null&&this.centered&&(this.accumulatedMeanGrads[o]={originalName:`${s}/mg`,variable:z(()=>kt(r).variable(i))});const a=Array.isArray(t)?t[o].tensor:t[s];if(a==null)return;const l=this.accumulatedMeanSquares[o].variable,c=this.accumulatedMoments[o].variable;z(()=>{const u=J(A(l,this.decay),A(Vt(a),1-this.decay));if(this.centered){const h=this.accumulatedMeanGrads[o].variable,d=J(A(h,this.decay),A(a,1-this.decay)),p=ct(A(a,this.learningRate),Ne(rt(u,J(Vt(d),this.epsilon)))),f=J(A(c,this.momentum),p);l.assign(u),h.assign(d),c.assign(f);const m=rt(r,f);r.assign(m)}else{const h=J(A(l,this.decay),A(Vt(a),1-this.decay)),d=J(A(c,this.momentum),ct(A(a,this.learningRate),Ne(J(h,this.epsilon))));l.assign(h),c.assign(d);const p=rt(r,d);r.assign(p)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&xt(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&xt(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&xt(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(t){t=await this.extractIterations(t);const e=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,e).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.accumulatedMoments=t.slice(e,e*2).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(e*2,e*3).map(o=>({originalName:o.name,variable:o.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,e){return new t(e.learningRate,e.decay,e.momentum,e.epsilon,e.centered)}}const PN=[mx,gx,xx,bx,yx,wx,zd];function zN(){for(const n of PN)X(n)}function Df(n,t,e,s){i(n),e=e??0,s=s??1,a(e,s);let o=0;const r=l=>(l.then(c=>{const u=e+ ++o/n.length*(s-e);return t(u),c}),l);function i(l){I(l!=null&&Array.isArray(l)&&l.length>0,()=>"promises must be a none empty array")}function a(l,c){I(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${l}`),I(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${c}`),I(c>=l,()=>`startFraction must be no more than endFraction, but got startFraction ${l} and endFraction ${c}`)}return Promise.all(n.map(r))}async function $x(n,t){t==null&&(t={});const e=t.fetchFunc==null?B().platform.fetch:t.fetchFunc,s=n.map(h=>e(h,t.requestInit,{isBinary:!0})),a=(t.onProgress==null?await Promise.all(s):await Df(s,t.onProgress,0,.5)).map(h=>h.arrayBuffer());return t.onProgress==null?await Promise.all(a):await Df(a,t.onProgress,.5,1)}function BN(n,t){var e;const s=t.fetchFunc==null?B().platform.fetch:t.fetchFunc;let o=0,r;return(e=t.onProgress)===null||e===void 0||e.call(t,0),new ReadableStream({pull:async i=>{for(var a;o<n.length;){r||(r=(await s(n[o],t.requestInit,{isBinary:!0})).body.getReader());const{done:l,value:c}=await r.read();if(l){o++,r=void 0,(a=t.onProgress)===null||a===void 0||a.call(t,o/n.length);continue}i.enqueue(c);return}i.close()}})}async function VN(n,t="",e,s){return WN(i=>$x(i,{requestInit:s}))(n,t,e)}function WN(n){return async(t,e="",s)=>{const o=t.map(()=>!1),r={},i=s!=null?s.map(()=>!1):[],a=[];if(t.forEach((p,f)=>{let m=0;p.weights.forEach(g=>{const x="quantization"in g?g.quantization.dtype:g.dtype,b=io[x]*U(g.shape),w=()=>{o[f]=!0,r[f]==null&&(r[f]=[]),r[f].push({manifestEntry:g,groupOffset:m,sizeBytes:b})};s!=null?s.forEach((y,$)=>{y===g.name&&(w(),i[$]=!0)}):w(),a.push(g.name),m+=b})}),!i.every(p=>p)){const p=s.filter((f,m)=>!i[m]);throw new Error(`Could not find weights in manifest with names: ${p.join(", ")}. 
Manifest JSON has weights with names: ${a.join(", ")}.`)}const l=o.reduce((p,f,m)=>(f&&p.push(m),p),[]),c=[];l.forEach(p=>{t[p].paths.forEach(f=>{const m=e+(e.endsWith("/")?"":"/")+f;c.push(m)})});const u=await n(c),h={};let d=0;return l.forEach(p=>{const f=t[p].paths.length,m=new ls(u.slice(d,d+f));r[p].forEach(x=>{const b=m.slice(x.groupOffset,x.groupOffset+x.sizeBytes),w=Ng(b,[x.manifestEntry]);for(const y in w)h[y]=w[y]}),d+=f}),h}}const UN="application/octet-stream",GN="application/json";class Bd{constructor(t,e){if(this.DEFAULT_METHOD="POST",e==null&&(e={}),this.weightPathPrefix=e.weightPathPrefix,this.weightUrlConverter=e.weightUrlConverter,e.fetchFunc!=null?(I(typeof e.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=e.fetchFunc):this.fetch=B().platform.fetch,I(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&I(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,e.requestInit!=null&&e.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=e.requestInit||{},this.loadOptions=e}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const e=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);e.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],o=mC(t,s);if(e.body.append("model.json",new Blob([JSON.stringify(o)],{type:GN}),"model.json"),t.weightData!=null){const i=ls.join(t.weightData);e.body.append("model.weights.bin",new Blob([i],{type:UN}),"model.weights.bin")}const r=await this.fetch(this.path,e);if(r.ok)return{modelArtifactsInfo:od(t),responses:[r]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${r.status}.`)}async loadModelJSON(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let e;try{e=await t.json()}catch{let i=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?i+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":i+=" Please make sure the server is serving valid JSON for this request.",new Error(i)}const s=e.modelTopology,o=e.weightsManifest;if(s==null&&o==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return e}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const t=await this.loadModelJSON();return xC(t,e=>this.loadWeights(e))}async loadStream(){const t=await this.loadModelJSON(),e=await this.getWeightUrls(t.weightsManifest),s=yf(t.weightsManifest),o=()=>BN(e,this.loadOptions);return Object.assign(Object.assign({},t),{weightSpecs:s,getWeightStream:o})}async getWeightUrls(t){const e=Array.isArray(this.path)?this.path[1]:this.path,[s,o]=HN(e),r=this.weightPathPrefix||s,i=[],a=[];for(const l of t)for(const c of l.paths)this.weightUrlConverter!=null?a.push(this.weightUrlConverter(c)):i.push(r+c+o);return this.weightUrlConverter&&i.push(...await Promise.all(a)),i}async loadWeights(t){const e=await this.getWeightUrls(t),s=yf(t),o=await $x(e,this.loadOptions);return[s,o]}}Bd.URL_SCHEME_REGEX=/^https?:\/\//;function HN(n){const t=n.lastIndexOf("/"),e=n.lastIndexOf("?"),s=n.substring(0,t),o=e>t?n.substring(e):"";return[s+"/",o]}function Ff(n){return n.match(Bd.URL_SCHEME_REGEX)!=null}const Cx=(n,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let e=!0;if(Array.isArray(n)?e=n.every(s=>Ff(s)):e=Ff(n),e)return Ix(n,t)}return null};Zt.registerSaveRouter(Cx);Zt.registerLoadRouter(Cx);function Ix(n,t){return new Bd(n,t)}function qN(n,t){return Ix(n,t)}function Sc(n,t){const e=n.shape.length,s=t.shape.length;if(e<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${e}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[s-1]>e)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[s-1]} vs. ${e}`);if(U(n.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${n.shape}.`);const o=t.shape,r=o[o.length-1];let i=1;for(let h=0;h<o.length-1;++h)i*=o[h];const a=n.shape,l=o.slice();l.pop();let c=1;for(let h=r;h<e;++h)c*=a[h],l.push(a[h]);const u=[...ut(n.shape).map(h=>h/c),1].slice(0,r);return[l,i,c,u]}const i6=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:Sc},Symbol.toStringTag,{value:"Module"}));const Ru=-2,jN=-1;function Vd(n,t,e){const s=n.shape.length;I(s===t.length,()=>`Error in slice${s}D: Length of begin ${t} must match the rank of the array (${s}).`),I(s===e.length,()=>`Error in slice${s}D: Length of size ${e} must match the rank of the array (${s}).`);for(let o=0;o<s;++o)I(t[o]+e[o]<=n.shape[o],()=>`Error in slice${s}D: begin[${o}] + size[${o}] (${t[o]+e[o]}) would overflow input.shape[${o}] (${n.shape[o]})`)}function KN(n){const t=[];let e=0;for(;n>0;)n&1&&t.push(e),n/=2,e++;return t}function Wd(n,t,e){const s=[];for(let o=0;o<n.length;o++)s[o]=Math.ceil((t[o]-n[o])/e[o]);return s}function vx(n,t,e,s){const o=[...n];for(let r=o.length;r<s.length;r++)o.push(1);for(let r=0;r<e;r++)r===0?o[t]=1:(o.splice(t,0,1),o.pop());return o}function kx(n,t,e){return e<=n?e:e-(t-1)}function Sx(n,t){const e=[];for(let s=0;s<n;s++)e.push(t+s);return e}function XN(n,t,e,s,o,r,i,a,l){const c=n.length;let u=new Array(c),h=new Array(c),d=new Array(c);if(t.length&&e>0){const p=t[0],f=e+1;u=Nx(i,p,f,s,n),h=Tx(a,p,f,o,n),d=vx(r,p,f,n)}else for(let p=0;p<c;p++)u[p]=Rx(i,s,r,n,p,l),h[p]=Ax(a,o,r,n,p,l),d[p]=Ex(r,p,l);return{begin:u,end:h,strides:d}}function Nx(n,t,e,s,o){const r=[...o],i=Sx(e,t);for(let a=0;a<r.length;a++)if(i.indexOf(a)>-1)r[a]=0;else{const l=kx(t,e,a);let c=s[l];n&1<<l&&(c=0),r[a]=c}return r}function Tx(n,t,e,s,o){const r=[...o],i=Sx(e,t);for(let a=0;a<r.length;a++)if(i.indexOf(a)>-1)r[a]=Number.MAX_SAFE_INTEGER;else{const l=kx(t,e,a);let c=s[l];n&1<<l&&(c=Number.MAX_SAFE_INTEGER),r[a]=c}for(let a=0;a<r.length;a++){const l=o[a];r[a]<0&&(r[a]+=l),r[a]=Ns(0,r[a],o[a])}return r}function Ex(n,t,e){let s=n[t];return(e&1<<t||s==null)&&(s=1),s}function Rx(n,t,e,s,o,r){let i=t[o];const a=e[o]||1;(n&1<<o||r&1<<o||i==null)&&(a>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const l=s[o];return i<0&&(i+=l),i=Ns(0,i,l-1),i}function Ax(n,t,e,s,o,r){let i=t[o];const a=e[o]||1;(n&1<<o||r&1<<o||i==null)&&(a>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const l=s[o];return i<0&&(i+=l),a>0?i=Ns(0,i,l):i=Ns(-1,i,l-1),i}function Ud(n,t,e){let s=e.length;for(let o=0;o<e.length;o++)if(e[o]>1){s=o;break}for(let o=s+1;o<e.length;o++)if(t[o]>0||e[o]!==n[o])return!1;return!0}function Gd(n,t){let e=n.length>0?n[n.length-1]:1;for(let s=0;s<n.length-1;s++)e+=n[s]*t[s];return e}function Nc(n,t,e){let s;const o=n.shape.length;typeof t=="number"?s=[t,...new Array(o-1).fill(0)]:t.length<o?s=t.concat(new Array(o-t.length).fill(0)):s=t.slice(),s.forEach(i=>{I(i!==-1,()=>"slice() does not support negative begin indexing.")});let r;return e==null?r=new Array(o).fill(-1):typeof e=="number"?r=[e,...new Array(o-1).fill(-1)]:e.length<o?r=e.concat(new Array(o-e.length).fill(-1)):r=e,r=r.map((i,a)=>i>=0?i:(I(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),n.shape[a]-s[a])),[s,r]}function Hd(n,t,e,s,o,r,i,a,l){let c;if(s==null?(c=new Array(t.length),c.fill(1)):c=s,i!=null&&(i&i-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let u=!1;const h={dims:c.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:e.slice(),strides:c.slice(),beginMask:o,endMask:r,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:l};for(let w=0;w<h.dims;w++)u&&(1<<w&a)!==0&&h.numAddAxisAfterEllipsis++,1<<w&i&&(u=!0);u||(h.ellipsisMask|=1<<h.dims,h.dims++);const d={dims:n.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};YN(h,d);let p=!0,f=!0,m=!0;const g=[],x=[];for(let w=0;w<n.length;++w){if(d.strides[w]===0)throw Error(`strides[${w}] must be non-zero`);const y=!!(d.shrinkAxisMask&1<<w),$=n[w];if($===-1){g.push(y?1:-1);continue}const v=[d.beginMask&1<<w,d.endMask&1<<w],N=[d.strides[w]>0?0:-1,d.strides[w]>0?$:$-1];if(y&&d.strides[w]<=0)throw Error("only stride 1 allowed on non-range indexing.");m=m&&d.strides[w]===1;const T=!!(d.beginMask&1<<w&&d.endMask&1<<w);if(d.beginValid&&d.endValid){if(y){const D=d.begin[w]<0?$+d.begin[w]:d.begin[w];if(d.begin[w]=D,d.end[w]=d.begin[w]+1,D<0||D>=$)throw Error(`slice index ${d.begin[w]} of dimension ${w} out of bounds.`)}else d.begin[w]=_f(d.begin[w],0,d.strides[w],$,v,N),d.end[w]=_f(d.end[w],1,d.strides[w],$,v,N);const C=d.strides[w]===1&&d.begin[w]===0&&d.end[w]===$;p=p&&C,f=f&&(w===0&&d.strides[w]===1||C)}else p=p&&d.strides[w]===1&&T,f=f&&(w===0&&d.strides[w]===1||T);let k,S=!1;if(d.beginValid&&d.endValid?(k=d.end[w]-d.begin[w],S=!0):y?(k=1,S=!0):T&&$>=0&&(d.strides[w]<0?k=-$:k=$,S=!0),S){let C;k===0||k<0!=d.strides[w]<0?C=0:C=Math.trunc(k/d.strides[w])+(k%d.strides[w]!==0?1:0),g.push(C)}else g.push(-1)}for(let w=0;w<d.finalShapeGatherIndices.length;++w){const y=d.finalShapeGatherIndices[w];y>=0?x.push(g[y]):y===Ru&&x.push(1)}return{finalShapeSparse:x.filter((w,y)=>d.finalShapeGatherIndices[y]!==Ru),finalShape:x,isIdentity:p,sliceDim0:f,isSimpleSlice:m,begin:d.begin,end:d.end,strides:d.strides}}function YN(n,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let e=0;t.beginValid=n.begin!=null,t.endValid=n.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let s=0;s<n.dims;s++)if(1<<s&n.ellipsisMask){const o=Math.min(t.dims-(n.dims-s)+1+n.numAddAxisAfterEllipsis,t.dims);for(;e<o;e++)t.begin[e]=0,t.end[e]=0,t.strides[e]=1,t.beginMask|=1<<e,t.endMask|=1<<e,t.finalShapeGatherIndices.push(e),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[e]=s}else if(1<<s&n.newAxisMask)t.finalShapeGatherIndices.push(Ru),t.finalShapeGatherIndicesSparse.push(-1);else{if(e===t.begin.length)throw Error(`Index out of range using input dim ${e}; input has only ${t.dims} dims, ${t.begin.length}.`);n.begin!=null&&(t.begin[e]=n.begin[s]),n.end!=null&&(t.end[e]=n.end[s]),t.strides[e]=n.strides[s],n.beginMask&1<<s&&(t.beginMask|=1<<e),n.endMask&1<<s&&(t.endMask|=1<<e),n.shrinkAxisMask&1<<s?(t.finalShapeGatherIndices.push(jN),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<e):(t.finalShapeGatherIndices.push(e),t.finalShapeGatherIndicesSparse.push(s)),t.inputShapeGatherIndicesSparse[e]=s,e++}}function _f(n,t,e,s,o,r){if(o[t])return e>0?r[t]:r[t+1&1];{const i=n<0?s+n:n;return i<r[0]?r[0]:i>r[1]?r[1]:i}}const ZN=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:Vd,computeFlatOffset:Gd,computeOutShape:Wd,getNormalizedAxes:XN,isSliceContinous:Ud,maskToAxes:KN,parseSliceParams:Nc,sliceInfo:Hd,startForAxis:Rx,startIndicesWithElidedDims:Nx,stopForAxis:Ax,stopIndicesWithElidedDims:Tx,stridesForAxis:Ex,stridesWithElidedDims:vx},Symbol.toStringTag,{value:"Module"}));class JN{static sgd(t){return new zd(t)}static momentum(t,e,s=!1){return new yx(t,e,s)}static rmsprop(t,e=.9,s=0,o=null,r=!1){return new wx(t,e,s,o,r)}static adam(t=.001,e=.9,s=.999,o=null){return new xx(t,e,s,o)}static adadelta(t=.001,e=.95,s=null){return new mx(t,e,s)}static adamax(t=.002,e=.9,s=.999,o=null,r=0){return new bx(t,e,s,o,r)}static adagrad(t,e=.1){return new gx(t,e)}}const _o=JN;const QN=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:n=>n();function Dx(){return new Promise(n=>QN(()=>n()))}function qd(n,t){const e=n[0].length;n.forEach((o,r)=>{I(o.length===e,()=>`Error in concat${e}D: rank of tensors[${r}] must be the same as the rank of the rest (${e})`)}),I(t>=0&&t<e,()=>`Error in concat${e}D: axis must be between 0 and ${e-1}.`);const s=n[0];n.forEach((o,r)=>{for(let i=0;i<e;i++)I(i===t||o[i]===s[i],()=>`Error in concat${e}D: Shape of tensors[${r}] (${o}) does not match the shape of the rest (${s}) along the non-concatenated axis ${r}.`)})}function Vn(n,t){const e=n[0].slice();for(let s=1;s<n.length;s++)e[t]+=n[s][t];return e}var vn;(function(n){n[n.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",n[n.VALUE_ROWIDS=1]="VALUE_ROWIDS",n[n.ROW_LENGTHS=2]="ROW_LENGTHS",n[n.ROW_SPLITS=3]="ROW_SPLITS",n[n.ROW_LIMITS=4]="ROW_LIMITS",n[n.ROW_STARTS=5]="ROW_STARTS"})(vn||(vn={}));function Fx(n,t,e){let s=new Array;if(e==null&&t==null)return s;if(t==null)for(;s.length<n+e.length;)s.push(-1);else s=t.slice();if(e==null)return s;if(n+e.length!==s.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${n+e.length}, but shape.rank = ${s.length}`);for(let o=1;o<e.length;++o){const r=e[o],i=s[s.length-e.length+o],a=s[i];if(r>=0)if(a>=0){if(a!==r)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${o+n}] = ${r} but shape[${o+n}] = ${a}`)}else s[i]=r}return s}function _x(n){const t={FIRST_DIM_SIZE:vn.FIRST_DIM_SIZE,VALUE_ROWIDS:vn.VALUE_ROWIDS,ROW_LENGTHS:vn.ROW_LENGTHS,ROW_SPLITS:vn.ROW_SPLITS,ROW_LIMITS:vn.ROW_LIMITS,ROW_STARTS:vn.ROW_STARTS},e=[];for(const s of n)if(s in t)e.push(t[s]);else break;return e}function Ox(n){return n.length===0?0:n[0]===vn.FIRST_DIM_SIZE?n.length-1:n.length}function Lx(n,t){if(n==null||t==null)return;const e=n.length,s=t.length;if(e>=s)throw new Error(`defaultValue.shape=${n} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${e} must be less than ragged tensor input flatValues.rank = ${s})`);for(let o=0;o<Math.min(e,s-1);++o){const r=n[o],i=t[o+1];if(r>=0&&i>=0&&r!==1&&r!==i)throw new Error(`defaultValue.shape=${n}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${o-n.length}] = ${r} but ragged tensor input.flatValues.shape[${o-n.length}] = ${i}`)}}const jd=30;function Tc(n){return n<=jd?n:Va(n,Math.floor(Math.sqrt(n)))}function Kd(n,t,e){const s=e*(typeof n=="number"?n:n[0]),o=t*(typeof n=="number"?n:n[1]);return[s,o]}function Zi(n,t,e,s=!0){let o=[];if(s)o=o.concat(t.slice(0)),o.push(n[0]/e),o=o.concat(n.slice(1));else{o=o.concat(n[0]);const r=t.length;for(let i=0;i<r;++i)o=o.concat([n[i+1]/t[i],t[i]]);o=o.concat(n.slice(r+1))}return o}function Ji(n,t,e=!0){const s=[];if(e){s.push(t);for(let o=t+1;o<n;++o)o<=2*t?(s.push(o),s.push(o-(t+1))):s.push(o)}else{const o=[],r=[];for(let i=1;i<n;++i)i>=t*2+1||i%2===1?r.push(i):o.push(i);s.push(...o),s.push(0),s.push(...r)}return s}function Qi(n,t,e,s=!0){const o=[];s?o.push(n[0]/e):o.push(n[0]*e);for(let r=1;r<n.length;++r)r<=t.length?s?o.push(t[r-1]*n[r]):o.push(n[r]/t[r-1]):o.push(n[r]);return o}function Xd(n,t){const e=[0];for(let s=0;s<t;++s)e.push(n[s][0]);return e}function Yd(n,t,e){const s=n.slice(0,1);for(let o=0;o<e;++o)s.push(n[o+1]-t[o][0]-t[o][1]);return s}const Ec=1.7580993408473768,Rc=1.0507009873554805;const Zd=.3275911,Jd=.254829592,Qd=-.284496736,tp=1.421413741,ep=-1.453152027,np=1.061405429;function os(n,t){if(n.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${n.length}, imag: ${t.length}.`);const e=new Float32Array(n.length*2);for(let s=0;s<e.length;s+=2)e[s]=n[s/2],e[s+1]=t[s/2];return e}function Mx(n){const t=new Float32Array(n.length/2),e=new Float32Array(n.length/2);for(let s=0;s<n.length;s+=2)t[s/2]=n[s],e[s/2]=n[s+1];return{real:t,imag:e}}function Px(n){const t=Math.ceil(n.length/4),e=new Float32Array(t),s=new Float32Array(t);for(let o=0;o<n.length;o+=4)e[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:e,imag:s}}function zx(n){const t=Math.floor(n.length/4),e=new Float32Array(t),s=new Float32Array(t);for(let o=2;o<n.length;o+=4)e[Math.floor(o/4)]=n[o],s[Math.floor(o/4)]=n[o+1];return{real:e,imag:s}}function sp(n,t){const e=n[t*2],s=n[t*2+1];return{real:e,imag:s}}function Bx(n,t,e,s){n[s*2]=t,n[s*2+1]=e}function Vx(n,t){const e=new Float32Array(n/2),s=new Float32Array(n/2);for(let o=0;o<Math.ceil(n/2);o++){const r=(t?2:-2)*Math.PI*(o/n);e[o]=Math.cos(r),s[o]=Math.sin(r)}return{real:e,imag:s}}function Wx(n,t,e){const s=(e?2:-2)*Math.PI*(n/t),o=Math.cos(s),r=Math.sin(s);return{real:o,imag:r}}const ou="->",tT=/->/g,Of=",",Lf="...";function op(n,t){n=n.replace(/\s/g,"");const e=(n.length-n.replace(tT,"").length)/ou.length;if(e<1)throw new Error("Equations without an arrow are not supported.");if(e>1)throw new Error(`Equation must contain exactly one arrow ("${ou}").`);const[s,o]=n.split(ou);I(s.indexOf(Lf)===-1,()=>`The ellipsis notation ("${Lf}") is not supported yet.`);const r=s.split(Of),i=r.length;if(t!==i)throw new Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const a=[];for(let d=0;d<o.length;++d){const p=o[d];if(!r.some(f=>f.indexOf(p)!==-1))throw new Error(`Output subscripts contain the label ${p} not present in the input subscripts.`);a.indexOf(p)===-1&&a.push(p)}for(let d=0;d<s.length;++d){const p=s[d];a.indexOf(p)===-1&&p!==Of&&a.push(p)}const l=new Array(r.length);for(let d=0;d<i;++d){if(new Set(r[d].split("")).size!==r[d].length)throw new Error(`Found duplicate axes in input component ${r[d]}. Support for duplicate axes in input is not implemented yet.`);l[d]=[];for(let p=0;p<r[d].length;++p)l[d].push(a.indexOf(r[d][p]))}const c=a.length,u=o.length,h=[];for(let d=u;d<c;++d)h.push(d);return{allDims:a,summedDims:h,idDims:l}}function rp(n,t){let e=new Array(n);e.fill(-1);for(let o=0;o<t.length;++o)e[t[o]]=o;const s=[];for(let o=0;o<n;++o)e[o]===-1&&s.push(o);return e=e.filter(o=>o!==-1),{permutationIndices:e,expandDims:s}}function ip(n,t,e){const s=new Array(n);for(let o=0;o<e.length;++o){const r=e[o].shape;for(let i=0;i<t[o].length;++i)s[t[o][i]]===void 0?s[t[o][i]]=r[i]:I(s[t[o][i]]===r[i],()=>`Expected dimension ${s[t[o][i]]} at axis ${i} of input shaped ${JSON.stringify(r)}, but got dimension ${r[i]}`)}}function ap(n,t){const e=n,s=[];let o=0;n.length===0&&e.push(-1),o=n.length+1;for(let i=0;i<o;++i)s.push([]);const r=[];for(let i=0;i<e.length;++i){const a=e[i],l=eT(t,a);for(const c of l)r.indexOf(c)===-1&&(s[i].push(c),r.push(c))}return{path:e,steps:s}}function lp(n){return n.every((t,e)=>t===e)}function eT(n,t){const e=[];for(let s=0;s<n.length;++s)(n[s].length===0||n[s].indexOf(t)!==-1||t===-1)&&e.push(s);return e}function cp(n,t,e=0){let s=[];if(typeof t=="number")I(n.shape[e]%t===0,()=>"Number of splits must evenly divide the axis."),s=new Array(t).fill(n.shape[e]/t);else{const o=t.reduce((i,a)=>(a===-1&&(i+=1),i),0);I(o<=1,()=>"There should be only one negative value in split array.");const r=t.indexOf(-1);if(r!==-1){const i=t.reduce((a,l)=>l>0?a+l:a);t[r]=n.shape[e]-i}I(n.shape[e]===t.reduce((i,a)=>i+a),()=>"The sum of sizes must match the size of the axis dimension."),s=t}return s}function Ux(n){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${n}`}function Gx(n,t){return`indices(${n}, 0) is invalid: ${t} < 0`}function Hx(n,t,e){return`indices(${n}, 0) is invalid: ${t} >= ${e}`}function qx(n,t){return`only one output dimension may be -1, not both ${n} and ${t}`}function jx(n,t){return`size ${n} must be non-negative, not ${t}`}function Kx(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Xx(n,t){const e=U(n),s=U(t);return`Input to reshape is a SparseTensor with ${e}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${n} outputShape= ${t}`}function Yx(n,t){const e=U(n),s=U(t);return`Input to reshape is a tensor with ${e} dense values, but the requested shape has ${s}. inputShape=${n} outputShape=${t}`}function Au(){return"segment ids must be >= 0"}function Zx(){return"segment ids are not increasing"}function Jx(n,t){return`Segment id ${n} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function Qx(n,t,e){return`Bad: indices[${n}] == ${t} out of range [0, ${e})`}function tb(n,t){let e=!1,s;for(n<=jd?(s=n,e=!0):s=Va(n,Math.floor(Math.sqrt(n)));!e;)s>t||s===n?e=!0:s=Va(n,s+1);return s}function eb(n,t,e){const s=[],o=n.length;for(let r=0;r<o;r++)r!==t?s.push(n[r]):s.push(e);return s}function up(n,t,e,s){const o=t.shape.length,r=n.shape.length;if(s!==0&&(s<-o||s>o))throw new Error(`Expect batchDims in the range of [-${o}, ${o}], but got ${s}`);if(s<0&&(s+=o),s>r)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${r}).`);if(e<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${e}).`);for(let h=0;h<s;++h)if(n.shape[h]!==t.shape[h])throw new Error(`x.shape[${h}]: ${n.shape[h]} should be equal to indices.shape[${h}]: ${t.shape[h]}.`);const i=n.shape[e],a=[];let l=1,c=1,u=1;for(let h=0;h<s;++h)a.push(n.shape[h]),l*=n.shape[h];for(let h=s;h<e;h++)a.push(n.shape[h]),c*=n.shape[h];for(let h=s;h<o;h++)a.push(t.shape[h]);for(let h=e+1;h<r;h++)a.push(n.shape[h]),u*=n.shape[h];return{batchSize:l,sliceSize:u,outerSize:c,dimSize:i,outputShape:a}}const nT=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:up,computeOutShape:eb,segOpComputeOptimalWindowSize:tb},Symbol.toStringTag,{value:"Module"}));function rs(n){try{return n.map(t=>ns(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function nb(n){return n.map(t=>es(t))}const sT=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:Jd,ERF_A2:Qd,ERF_A3:tp,ERF_A4:ep,ERF_A5:np,ERF_P:Zd,PARALLELIZE_THRESHOLD:jd,get RowPartitionType(){return vn},SELU_SCALE:Rc,SELU_SCALEALPHA:Ec,applyActivation:Fd,assertAndGetBroadcastShape:gt,assertAxesAreInnerMostDims:Ie,assertParamsConsistent:qd,assignToTypedArray:Bx,axesAreInnerMostDims:hd,calculateShapes:Bs,checkEinsumDimSizes:ip,checkPadOnDimRoundingMode:Ge,combineLocations:Gg,combineRaggedTensorToTensorShapes:Fx,complexWithEvenIndex:Px,complexWithOddIndex:zx,computeConv2DInfo:Ce,computeConv3DInfo:Ms,computeDefaultPad:rd,computeDilation2DInfo:Hi,computeOptimalWindowSize:Tc,computeOutAndReduceShapes:me,computeOutShape:Vn,computePool2DInfo:xn,computePool3DInfo:cs,convertConv2DDataFormat:us,decodeEinsumEquation:op,eitherStridesOrDilationsAreOne:Te,expandShapeToKeepDim:ee,exponent:Wx,exponents:Vx,fromStringArrayToUint8:nb,fromUint8ToStringArray:rs,getAxesPermutation:Xt,getBroadcastDims:po,getComplexWithIndex:sp,getEinsumComputePath:ap,getEinsumPermutation:rp,getFusedBiasGradient:Dd,getFusedDyActivation:Ad,getImageCenter:Kd,getInnerMostAxes:ne,getPermuted:Ji,getRaggedRank:Ox,getReductionAxes:re,getReshaped:Zi,getReshapedPermuted:Qi,getRowPartitionTypesHelper:_x,getSliceBeginCoords:Xd,getSliceSize:Yd,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Ux,getSparseFillEmptyRowsNegativeIndexErrorMessage:Gx,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:Hx,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:Kx,getSparseReshapeInputOutputMismatchErrorMessage:Yx,getSparseReshapeInputOutputMultipleErrorMessage:Xx,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:qx,getSparseReshapeNegativeOutputDimErrorMessage:jx,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Qx,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:Au,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Zx,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Jx,getUndoAxesPermutation:Ps,isIdentityPermutation:lp,log:F$,mergeRealAndImagArrays:os,prepareAndValidate:Sc,prepareSplitSize:cp,segment_util:nT,shouldFuse:_d,slice_util:ZN,splitRealAndImagArrays:Mx,stridesOrDilationsArePositive:uo,tupleValuesAreOne:co,upcastType:Ke,validateDefaultValueShape:Lx,validateInput:ix,validateUpdateShape:Ed,warn:Qe},Symbol.toStringTag,{value:"Module"}));zN();const sb={kernelName:gl,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,Yi(st(e,"float32"),-1))}}};const oT={kernelName:Xr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Vt(st(e,"float32")),o=Ne(rt(mt(1),s));return Ht(ct(n,o))}}}};const rT={kernelName:Yr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Ne(rt(Vt(st(e,"float32")),1));return ct(n,s)}}}};const iT={kernelName:tr,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{let a=n;const l=re(e.shape,o);return l.length>0&&(a=it(a,l)),L(a,e.shape)},b:()=>{let a=n;const l=re(s.shape,o);return l.length>0&&(a=it(a,l)),L(a,s.shape)}}}};const aT={kernelName:sh,saveAllInputs:!0,gradFunc:(n,t)=>{const e={};return t.forEach((s,o)=>{e[o]=()=>n.clone()}),e}};const lT={kernelName:xl,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>kt(e)}}};const cT={kernelName:bl,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>kt(e)}}};const uT={kernelName:Zr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,Ne(rt(mt(1),Vt(st(e,"float32")))))}}};const hT={kernelName:Jr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Ne(J(mt(1),Vt(st(e,"float32"))));return ct(n,s)}}}};const dT={kernelName:ei,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{const a=J(Vt(e),Vt(s));let l=A(n,ct(s,a));const c=re(e.shape,o);return c.length>0&&(l=it(l,c)),L(l,e.shape)},b:()=>{const a=J(Vt(e),Vt(s));let l=Ht(A(n,ct(e,a)));const c=re(s.shape,o);return c.length>0&&(l=it(l,c)),L(l,s.shape)}}}};const pT={kernelName:Qr,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,J(Vt(st(e,"float32")),1))}}};const fT={kernelName:ti,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,rt(mt(1),Vt(st(e,"float32"))))}}};function mT(n,t,e,s,o,r){const i=E(n,"dy","avgPool3dGrad"),a=E(t,"input","avgPool3dGrad");let l=i,c=a,u=!1;a.rank===4&&(u=!0,l=L(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]]),c=L(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]])),I(l.rank===5,()=>`Error in avgPool3dGrad: dy must be rank 5 but got rank ${l.rank}.`),I(c.rank===5,()=>`Error in avgPool3dGrad: input must be rank 5 but got rank ${c.rank}.`),Ge("avgPool3dGrad",o,r);const h={dy:l,input:c},d={filterSize:e,strides:s,pad:o,dimRoundingMode:r},p=R.runKernel(ah,h,d);return u?L(p,[p.shape[1],p.shape[2],p.shape[3],p.shape[4]]):p}const gT=_({avgPool3dGrad_:mT});const xT={kernelName:wl,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{filterSize:o,strides:r,pad:i,dimRoundingMode:a}=e;return{x:()=>gT(n,s,o,r,i,a)}}};function bT(n,t,e,s,o){const r=E(n,"dy","avgPoolGrad"),i=E(t,"input","avgPoolGrad");I(i.rank===r.rank,()=>`Rank of input (${i.rank}) does not match rank of dy (${r.rank})`);let a=i,l=r,c=!1;i.rank===3&&(c=!0,a=L(i,[1,i.shape[0],i.shape[1],i.shape[2]]),l=L(r,[1,r.shape[0],r.shape[1],r.shape[2]])),I(l.rank===4,()=>`Error in avgPoolGrad: dy must be rank 4 but got rank ${l.rank}.`),I(a.rank===4,()=>`Error in avgPoolGrad: input must be rank 4 but got rank ${a.rank}.`);const u={dy:l,input:a},h={filterSize:e,strides:s,pad:o},d=R.runKernel(ih,u,h);return c?L(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const yT=_({avgPoolGrad_:bT});const wT={kernelName:yl,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{filterSize:o,strides:r,pad:i}=e;return{x:()=>yT(n,s,o,r,i)}}};const $T={kernelName:$l,inputsToSave:["a","b"],gradFunc:(n,t,e)=>{const[s,o]=t,{transposeA:r,transposeB:i}=e;return!r&&!i?{a:()=>Et(n,o,!1,!0),b:()=>Et(s,n,!0,!1)}:!r&&i?{a:()=>Et(n,o,!1,!1),b:()=>Et(n,s,!0,!1)}:r&&!i?{a:()=>Et(o,n,!1,!0),b:()=>Et(s,n,!1,!1)}:{a:()=>Et(o,n,!0,!0),b:()=>Et(n,s,!0,!0)}}};const CT={kernelName:Cl,gradFunc:(n,t,e)=>{const{blockShape:s,crops:o}=e;return{x:()=>$d(n,s,o)}}};const IT={kernelName:E$,gradFunc:(n,t,e)=>{const s=e,o=s.inputShape,r=s.shape,i=Array.from(r);for(let l=o.length-1;l>=0;l--)if(o[l]===r[l])i[l]=1;else if(o[l]!==1)throw new Error(`broadcastTo(): [${o}] cannot be broadcast to [${r}].`);const a=[];for(let l=0;l<i.length;l++)i[l]>1&&a.push(l);return{x:()=>it(n,a,!0)}}};const vT={kernelName:ni,gradFunc:n=>({x:()=>n.clone()})};const kT={kernelName:si,gradFunc:n=>({x:()=>kt(n)})};const ST={kernelName:oi,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{clipValueMin:o,clipValueMax:r}=e;return{x:()=>Se(Gn(So(s,o),nr(s,r)),n,kt(n))}}};const NT={kernelName:Il,inputsToSave:["x"],gradFunc:sb.gradFunc};const TT={kernelName:vl,saveAllInputs:!0,gradFunc:(n,t,e)=>{const s=t.map(l=>l.shape),{axis:o}=e,r=$t(o,t[0].shape)[0],i=s.map(l=>l[r]);return sn(n,i,r).map(l=>()=>l)}};const ET={kernelName:kl,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,{dilations:r,strides:i,pad:a,dataFormat:l}=e;return I(co(r),()=>`Error in gradient of conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${r}'`),{x:()=>ld(s.shape,n,o,i,a,l),filter:()=>Rd(s,n,o.shape,i,a,l)}}};const RT={kernelName:Sl,inputsToSave:["dy","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,{strides:r,pad:i,dataFormat:a,dimRoundingMode:l}=e;return{dy:()=>ho(n,o,r,i,a,1,l),filter:()=>Rd(n,s,o.shape,r,i,a,l)}}};function AT(n,t,e,s,o){let r=n;n.rank===4&&(r=L(n,[1,n.shape[0],n.shape[1],n.shape[2],n.shape[3]]));let i=t;i.rank===4&&(i=L(t,[1,t.shape[0],t.shape[1],t.shape[2],t.shape[3]])),I(r.rank===5,()=>`Error in conv3dDerFilter: input must be rank 5, but got shape ${r.shape}.`),I(i.rank===5,()=>`Error in conv3dDerFilter: dy must be rank 5, but got shape ${i.shape}.`),I(e.length===5,()=>`Error in conv3dDerFilter: filterShape must be length 5, but got ${e}.`),I(r.shape[4]===e[3],()=>`Error in conv3dDerFilter: depth of input ${r.shape[4]}) must match input depth in filter (${e[3]}.`),I(i.shape[4]===e[4],()=>`Error in conv3dDerFilter: depth of dy (${i.shape[4]}) must match output depth for filter (${e[4]}).`);const a={x:r,dy:i},l={strides:s,pad:o,filterShape:e};return R.runKernel(dh,a,l)}const DT=_({conv3DBackpropFilter_:AT});const FT={kernelName:Nl,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const{dilations:s,strides:o,pad:r}=e;I(co(s),()=>`Error in gradient of conv3D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${s}'`);const[i,a]=t;return{x:()=>Bg(i.shape,n,a,o,r),filter:()=>DT(i,n,a.shape,o,r)}}};const _T={kernelName:ri,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(Ht(ex(st(e,"float32"))),n)}}};const OT={kernelName:ii,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(nx(st(e,"float32")),n)}}};const LT={kernelName:Tl,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o,exclusive:r,reverse:i}=e;return{x:()=>{const a=Xt([o],s.rank);let l=Wg(n,o,r,!i);return a!=null&&(l=vt(l,a)),l}}}};const MT={kernelName:El,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const{dilations:s,strides:o,pad:r,dimRoundingMode:i}=e,a=s??[1,1];I(co(a),()=>`Error in gradient of depthwiseConv2dNative: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[l,c]=t;return I(l.rank===4,()=>`Error in gradient of depthwiseConv2dNative: input must be rank 4, but got rank ${l.rank}.`),I(c.rank===4,()=>`Error in gradient of depthwiseConv2dNative: filter must be rank 4, but got rank ${c.rank}.`),I(l.shape[3]===c.shape[2],()=>`Error in gradient of depthwiseConv2d: number of input channels (${l.shape[3]}) must match the inChannels dimension in filter ${c.shape[2]}.`),I(Te(o,a),()=>`Error in gradient of depthwiseConv2d: Either strides or dilations must be  1. Got strides ${o} and dilations '${a}'.`),Ge("depthwiseConv2d",r,i),{x:()=>r2(l.shape,n,c,o,r,a,i),filter:()=>s2(l,n,c.shape,o,r,a,i)}}};const PT={kernelName:Rl,inputsToSave:["x","filter"],gradFunc:(n,t,e)=>{const[s,o]=t,r={x:s,filter:o,dy:n},i={x:s,filter:o,dy:n};return{x:()=>R.runKernel(fu,r,e),filter:()=>R.runKernel(mu,i,e)}}};const zT={kernelName:li,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t,s={dy:n,y:e};return{x:()=>R.runKernel($h,s)}}};const BT={kernelName:ci,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t,s=A(rn(Ht(Vt(e))),2/Math.sqrt(Math.PI));return{x:()=>A(n,s)}}};const VT={kernelName:ui,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,e)}}};const WT={kernelName:Dl,inputsToSave:["input"],gradFunc:(n,t)=>{const[e]=t;return{input:()=>L(n,e.shape)}}};const UT={kernelName:hi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,rn(e))}}};const GT={kernelName:di,gradFunc:n=>({x:()=>kt(n)})};const HT={kernelName:pi,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{const a=ct(n,st(s,"float32")),l=re(e.shape,o);return l.length>0?L(it(a,l),e.shape):a},b:()=>{let a=A(n,st(e,"float32"));const l=re(s.shape,o);l.length>0&&(a=L(it(a,l),s.shape));const c=Vt(s);return Ht(ct(a,st(c,"float32")))}}}};const qT={kernelName:Fl,inputsToSave:["x","mean","variance","scale"],gradFunc:(n,t,e)=>{const{varianceEpsilon:s}=e,[o,r,i,a]=t,l=a??mt(1),c=re(r.shape,o.shape),u=[];if(r.rank===1){for(let y=0;y<o.shape.length-1;++y)u.push(o.shape[y]);u.push(1)}const h=rt(o,r),d=A(n,l),p=Jg(J(i,mt(s))),f=A(A(A(p,p),p),mt(-.5));return{x:()=>r.rank===1?L(A(A(n,kn(L(p,[1,1,1,r.shape[0]]),u)),l),o.shape):L(A(A(n,p),l),o.shape),mean:()=>{let y=A(A(p,mt(-1)),d);return r.rank===1&&(y=it(y,c)),L(y,r.shape)},variance:()=>{let y=A(A(f,h),d);return r.rank===1&&(y=it(y,c)),L(y,r.shape)},scale:()=>{const y=A(h,p);let $=A(n,y);return r.rank===1&&($=it($,c)),L($,r.shape)},offset:()=>{let y=n;return r.rank===1&&(y=it(y,c)),L(y,r.shape)}}}};const jT={kernelName:_l,inputsToSave:["x","indices"],gradFunc:(n,t,e)=>{const[s,o]=t,{axis:r,batchDims:i}=e,a=$t(r,s.shape)[0],l=(c,u,h)=>()=>{const d=c.shape,p=u.size,f=d.slice(0,a),m=f.length,g=d.slice(r,d.length).slice(1),x=g.length,b=Mf(0,m),w=Mf(m+1,m+1+x),y=Pf([f,[p],g]),$=L(h,y),v=L(u,[p]),N=Pf([[m],b,w]),T=vt($,N);let k=lx(T,v,c.shape[a]);const S=Ps(N);return k=vt(k,S),k};if(i===1){const c=s.shape[0],u=s.split(c,0);return{x:()=>Hn(u.map((p,f)=>l(p,o.slice(f,1),n.slice(f,1))())).reshape(s.shape),indices:()=>o}}else return{x:l(s,o,n),indices:()=>o}}};function Mf(n,t){const e=[];for(let s=n;s<t;++s)e.push(s);return e}function Pf(n){const t=[];for(let e=0;e<n.length;++e)for(let s=0;s<n[e].length;++s)t.push(n[e][s]);return t}const KT={kernelName:fi,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>kt(e),b:()=>kt(s)}}};const XT={kernelName:mi,gradFunc:n=>({x:()=>st(n,"float32")})};const YT={kernelName:gi,gradFunc:n=>({x:()=>kt(n)})};const ZT={kernelName:xi,gradFunc:n=>({x:()=>kt(n)})};const JT={kernelName:bi,gradFunc:n=>({x:()=>kt(n)})};const QT={kernelName:Ll,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{alpha:o}=e,r=Ze(s,0);return{x:()=>Se(r,n,A(n,o))}}};const tE={kernelName:wi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,J(e,1))}}};const eE={kernelName:yi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,st(e,"float32"))}}};const nE={kernelName:A$,inputsToSave:[],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e;return{logits:()=>{const i=rn(s);return rt(n,A(it(n,o,!0),i))}}}};function sE(n,t,e,s=5,o=1,r=1,i=.5){const a={x:n,y:t,dy:e},l={depthRadius:s,bias:o,alpha:r,beta:i};return R.runKernel(Th,a,l)}const oE=_({localResponseNormalizationBackprop_:sE});const rE={kernelName:Wl,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{depthRadius:r,bias:i,alpha:a,beta:l}=e;return{x:()=>oE(s,o,n,r,i,a,l)}}};function ob(n,t,e,s){return t.rank<e.rank&&(t=L(t,ee(t.shape,s))),n.rank<e.rank&&(n=L(n,ee(n.shape,s))),{x:()=>A(n,st(fn(e,t),n.dtype))}}const zf={kernelName:Ul,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const s=e,{reductionIndices:o}=s,r=t[0],i=t[1],a=$t(o,r.shape),l=ob(n,i,r,a);return{x:()=>l.x()}}};const iE={kernelName:$i,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>A(n,st(So(e,s),"float32")),b:()=>A(n,st(qa(e,s),"float32"))}}};function aE(n,t,e,s,o,r,i){const a=E(n,"dy","maxPool3dGrad"),l=E(t,"input","maxPool3dGrad"),c=E(e,"output","maxPool3dGrad");let u=a,h=l,d=c,p=!1;l.rank===4&&(p=!0,u=L(a,[1,a.shape[0],a.shape[1],a.shape[2],a.shape[3]]),h=L(l,[1,l.shape[0],l.shape[1],l.shape[2],l.shape[3]]),d=L(c,[1,c.shape[0],c.shape[1],c.shape[2],c.shape[3]])),I(u.rank===5,()=>`Error in maxPool3dGrad: dy must be rank 5 but got rank ${u.rank}.`),I(h.rank===5,()=>`Error in maxPool3dGrad: input must be rank 5 but got rank ${h.rank}.`),I(d.rank===5,()=>`Error in maxPool3dGrad: output must be rank 5 but got rank ${d.rank}.`),Ge("maxPool3dGrad",r,i);const f={dy:u,input:h,output:d},m={filterSize:s,strides:o,pad:r,dimRoundingMode:i},g=R.runKernel(Rh,f,m);return p?L(g,[g.shape[1],g.shape[2],g.shape[3],g.shape[4]]):g}const lE=_({maxPool3dGrad_:aE});const cE={kernelName:Hl,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=e;return{x:()=>lE(n,s,o,r,i,a,l)}}};function uE(n,t,e,s,o,r,i){const a=E(n,"dy","maxPoolGrad"),l=E(t,"input","maxPoolGrad"),c=E(e,"output","maxPoolGrad");I(l.rank===a.rank,()=>`Rank of input (${l.rank}) does not match rank of dy (${a.rank})`),I(a.rank===4,()=>`Error in maxPoolGrad: dy must be rank 4 but got rank ${a.rank}.`),I(l.rank===4,()=>`Error in maxPoolGrad: input must be rank 4 but got rank ${l.rank}.`),Ge("maxPoolGrad",r,i);const u={dy:a,input:l,output:c},h={filterSize:s,strides:o,pad:r,dimRoundingMode:i};return R.runKernel(Eh,u,h)}const hE=_({maxPoolGrad_:uE});const dE={kernelName:Gl,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s,o]=t,{filterSize:r,strides:i,pad:a}=e;return{x:()=>hE(n,s,o,r,i,a)}}};const pE={kernelName:ql,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e,r=$t(o,s.shape),a=me(s.shape,r)[1],l=U(a);return{x:()=>{const u=s.shape.slice();r.forEach(p=>{u[p]=1});const h=L(n,u);return ct(A(h,hs(s.shape,"float32")),l)}}}};const fE={kernelName:jl,inputsToSave:["x"],outputsToSave:[!0],gradFunc:(n,t,e)=>{const s=e,{axis:o}=s,[r,i]=t,a=$t(o,r.shape),l=ob(n,i,r,a);return{x:()=>l.x()}}};const mE={kernelName:Ci,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t;return{a:()=>A(n,st(nr(e,s),"float32")),b:()=>A(n,st(Ze(e,s),"float32"))}}};const gE={kernelName:Kl,inputsToSave:["x"],gradFunc:(n,t,e)=>{const s=t[0],{paddings:o}=e,r=o.map(i=>i[0]);return{x:()=>_t(n,r,s.shape)}}};const xE={kernelName:Ii,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{const a=re(e.shape,o);return a.length>0?L(it(n,a),e.shape):n},b:()=>{const a=A(n,Ht(vc(ct(e,s)))),l=re(s.shape,o);return l.length>0?L(it(a,l),s.shape):a}}}};const bE={kernelName:vi,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{const a=A(n,st(s,"float32")),l=re(e.shape,o);return l.length>0?L(it(a,l),e.shape):a},b:()=>{const a=A(n,st(e,"float32")),l=re(s.shape,o);return l.length>0?L(it(a,l),s.shape):a}}}};const yE={kernelName:Xl,gradFunc:n=>({x:()=>Ht(n)})};const wE={kernelName:Jl,inputsToSave:["indices"],gradFunc:(n,t)=>{const e=t[0];return{indices:()=>pe(e.shape,"float32")}}};const $E={kernelName:Zl,gradFunc:n=>({x:()=>kt(n)})};const CE={kernelName:Ql,saveAllInputs:!0,gradFunc:(n,t,e)=>{const{axis:s}=e;return xo(n,s).map(r=>()=>r)}};const Bf={kernelName:tc,inputsToSave:["x"],gradFunc:(n,t,e)=>{const s=t[0],{paddings:o}=e,r=o.map(i=>i[0]);return{x:()=>_t(n,r,s.shape)}}};const IE={kernelName:ki,inputsToSave:["a","b"],outputsToSave:[!0],gradFunc:(n,t)=>{const[e,s,o]=t,r=e,i=s,a=gt(r.shape,i.shape);return{a:()=>{const u=st(i,"float32");let h=A(n,A(u,fo(r,rt(u,mt(1)))));const d=re(r.shape,a);return d.length>0&&(h=it(h,d)),L(h,r.shape)},b:()=>{const u=Ze(r,0),h=Se(u,mn(r),kt(r));let d=A(n,A(o,h));const p=re(i.shape,a);return p.length>0&&(d=it(d,p)),L(d,i.shape)}}}};const vE={kernelName:ec,inputsToSave:["x","alpha"],gradFunc:(n,t)=>{const[e,s]=t,o=Ze(e,0);return{x:()=>Se(o,n,A(n,s)),alpha:()=>{let r=Se(o,kt(n),A(n,e));const i=re(s.shape,n.shape);return i.length>0&&(r=it(r,i)),L(r,s.shape)}}}};function kE(n,t,e){const s=n.shape.slice();s[e]=1;const o=L(t,s),r=Tu(n,e,!0,!1),i=Tu(n,e,!0,!0),a=A(r,i);return A(o,a)}function SE(n,t,e){const s=n.shape.length,o=s-e.length,r=Xt(e,s);let i=n;r!=null&&(i=vt(n,r));const a=i.shape.slice(),c=a.splice(s-e.length,e.length).reduce((d,p)=>d*p,1);a.push(c);const u=i.reshape(a);let h=kE(u,t,o);if(h=h.reshape(i.shape),r!=null){const d=Ps(r);h=vt(h,d)}return h}const NE={kernelName:nc,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{axis:o}=e;let r=[];return o==null?r=s.shape.map((i,a)=>a):typeof o=="number"?r=[o]:r=o,{x:()=>SE(s,n,r)}}};const TE={kernelName:ai,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{const a=ct(n,st(s,"float32")),l=re(e.shape,o);return l.length>0?L(it(a,l),e.shape):a},b:()=>{let a=A(n,st(e,"float32"));const l=re(s.shape,o);l.length>0&&(a=L(it(a,l),s.shape));const c=Vt(s);return Ht(ct(a,st(c,"float32")))}}}};const EE={kernelName:Si,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,Ht(Vt(e)))}}};const RE={kernelName:Ti,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t,s=A(nr(e,6),Yi(e));return{x:()=>A(n,st(s,"float32"))}}};const AE={kernelName:Ni,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,st(Yi(e),"float32"))}}};const DE={kernelName:sc,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>L(n,e.shape)}}};const FE={kernelName:rc,inputsToSave:["images"],gradFunc:(n,t,e)=>{const[s]=t,o={dy:n,images:s};return{images:()=>R.runKernel(Mh,o,e)}}};const _E={kernelName:oc,inputsToSave:["images"],gradFunc:(n,t,e)=>{const[s]=t,o={dy:n,images:s};return{images:()=>R.runKernel(Lh,o,e)}}};const OE={kernelName:ic,gradFunc:(n,t,e)=>{const{dims:s}=e,o=$t(s,n.shape);return{x:()=>go(n,o)}}};const LE={kernelName:Ei,gradFunc:n=>({x:()=>kt(n)})};const ME={kernelName:Ri,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>Ht(ct(n,A(fo(e,1.5),2)))}}};const PE={kernelName:ac,inputsToSave:["condition"],gradFunc:(n,t)=>{const[e]=t;return{condition:()=>st(kt(e),"float32"),t:()=>A(n,st(e,n.dtype)),e:()=>A(n,st(xd(e),n.dtype))}}};const zE={kernelName:Ai,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>{const s=Ze(e,mt(0)),o=mt(Ec),r=mt(Rc),i=A(n,r),a=A(A(n,o),rn(st(e,"float32")));return Se(s,i,a)}}}};const BE={kernelName:Oi,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,A(e,rt(mt(1),e)))}}};const VE={kernelName:_i,gradFunc:n=>({x:()=>kt(n)})};const WE={kernelName:Di,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(cd(st(e,"float32")),n)}}};const UE={kernelName:Fi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(Vg(st(e,"float32")),n)}}};const GE={kernelName:lc,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{begin:o,size:r}=e,i=s.shape,[a,l]=Nc(s,o,r),c=[];for(let u=0;u<n.rank;u++)c.push([a[u],i[u]-a[u]-l[u]]);return{x:()=>wd(n,c)}}};const HE={kernelName:dc,outputsToSave:[!0],gradFunc:(n,t,e)=>{const[s]=t,{dim:o}=e,r=!0,i=A(n,s);return{logits:()=>rt(i,A(it(i,[o],r),s))}}};const qE={kernelName:Li,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,er(e))}}};const Vf={kernelName:uc,gradFunc:(n,t,e)=>{const{blockShape:s,paddings:o}=e;return{x:()=>ad(n,s,o)}}};const Wf={kernelName:hc,gradFunc:(n,t,e)=>{const{axis:s}=e;return{x:()=>ke(n,s)}}};const jE={kernelName:Mi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,A(Ne(st(e,"float32")),2))}}};const KE={kernelName:Wh,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(n,A(st(e,"float32"),2))}}};const XE={kernelName:Pi,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=mt(2);return{a:()=>A(n,A(o,rt(e,s))),b:()=>A(n,A(o,rt(s,e)))}}};const YE={kernelName:Ui,gradFunc:n=>({x:()=>kt(n)})};const ZE={kernelName:zi,inputsToSave:["a","b"],gradFunc:(n,t)=>{const[e,s]=t,o=gt(e.shape,s.shape);return{a:()=>{let a=n;const l=re(e.shape,o);return l.length>0&&(a=it(a,l)),L(a,e.shape)},b:()=>{let a=n;const l=re(s.shape,o);return l.length>0&&(a=it(a,l)),L(Ht(a),s.shape)}}}};const JE={kernelName:cc,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,o=s.shape.slice(),{axis:r}=e;$t(r,s.shape).forEach(c=>{o[c]=1});const a=L(n,o),l=A(a,hs(s.shape,"float32"));return{x:()=>l}}};const QE={kernelName:Bi,inputsToSave:["x"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>ct(n,Vt(cd(e)))}}};const tR={kernelName:Vi,outputsToSave:[!0],gradFunc:(n,t)=>{const[e]=t;return{x:()=>A(rt(mt(1),Vt(e)),n)}}};const eR={kernelName:Wi,inputsToSave:["x"],gradFunc:(n,t,e)=>{const[s]=t,{reps:o}=e;return{x:()=>{let i=kt(s);if(s.rank===1)for(let a=0;a<o[0];++a)i=J(i,_t(n,[a*s.shape[0]],[s.shape[0]]));else if(s.rank===2)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)i=J(i,_t(n,[a*s.shape[0],l*s.shape[1]],[s.shape[0],s.shape[1]]));else if(s.rank===3)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)for(let c=0;c<o[2];++c)i=J(i,_t(n,[a*s.shape[0],l*s.shape[1],c*s.shape[2]],[s.shape[0],s.shape[1],s.shape[2]]));else if(s.rank===4)for(let a=0;a<o[0];++a)for(let l=0;l<o[1];++l)for(let c=0;c<o[2];++c)for(let u=0;u<o[3];++u)i=J(i,_t(n,[a*s.shape[0],l*s.shape[1],c*s.shape[2],u*s.shape[3]],[s.shape[0],s.shape[1],s.shape[2],s.shape[3]]));else throw new Error(`Gradient for tile operation is not implemented for rank-${s.rank} tensors yet.`);return i}}}};const nR={kernelName:zo,gradFunc:(n,t,e)=>{const s=e,{perm:o}=s,r=Ps(o);return{x:()=>vt(n,r)}}};const sR={kernelName:fc,gradFunc:(n,t,e)=>{const s=e,{axis:o}=s;return{value:()=>Hn(n,o)}}};const oR={kernelName:mc,inputsToSave:["segmentIds"],gradFunc:(n,t)=>{const[e]=t;return{x:()=>rR(n,e)}}};function rR(n,t){const e=zs(t,kt(t)),s=dd(n,e);let o=So(t,mt(0,"int32"));const r=s.rank-o.rank;for(let a=0;a<r;++a)o=Me(o,a+1);o=Gn(o,hs(s.shape,"bool"));const i=kt(s);return Se(o,s,i)}const iR={kernelName:gc,gradFunc:n=>({x:()=>kt(n)})};const aR=[sb,oT,rT,iT,aT,lT,cT,uT,hT,dT,pT,fT,xT,wT,$T,CT,IT,vT,kT,ST,NT,TT,RT,ET,FT,_T,OT,LT,MT,PT,TE,zT,BT,VT,WT,UT,HT,GT,qT,jT,KT,XT,YT,ZT,JT,QT,tE,eE,nE,rE,zf,zf,iE,cE,dE,pE,fE,mE,gE,xE,bE,yE,wE,$E,CE,Bf,Bf,IE,vE,NE,EE,RE,AE,DE,FE,_E,OE,LE,ME,PE,zE,BE,VE,WE,UE,GE,HE,qE,Vf,Vf,Wf,Wf,jE,XE,KE,YE,ZE,JE,QE,tR,eR,nR,sR,oR,iR];for(const n of aR)_$(n);G().prototype.abs=function(){return this.throwIfDisposed(),he(this)};G().prototype.acos=function(){return this.throwIfDisposed(),KC(this)};G().prototype.acosh=function(){return this.throwIfDisposed(),YC(this)};G().prototype.add=function(n){return this.throwIfDisposed(),J(this,n)};G().prototype.all=function(n,t){return this.throwIfDisposed(),Mg(this,n,t)};G().prototype.any=function(n,t){return this.throwIfDisposed(),Su(this,n,t)};G().prototype.argMax=function(n){return this.throwIfDisposed(),Fr(this,n)};G().prototype.argMin=function(n){return this.throwIfDisposed(),eI(this,n)};G().prototype.asScalar=function(){return this.throwIfDisposed(),I(this.size===1,()=>"The array must have only 1 element."),L(this,[])};G().prototype.asType=function(n){return this.throwIfDisposed(),st(this,n)};G().prototype.as1D=function(){return this.throwIfDisposed(),L(this,[this.size])};G().prototype.as2D=function(n,t){return this.throwIfDisposed(),L(this,[n,t])};G().prototype.as3D=function(n,t,e){return this.throwIfDisposed(),L(this,[n,t,e])};G().prototype.as4D=function(n,t,e,s){return this.throwIfDisposed(),L(this,[n,t,e,s])};G().prototype.as5D=function(n,t,e,s,o){return this.throwIfDisposed(),L(this,[n,t,e,s,o])};G().prototype.asin=function(){return this.throwIfDisposed(),sI(this)};G().prototype.asinh=function(){return this.throwIfDisposed(),rI(this)};G().prototype.atan=function(){return this.throwIfDisposed(),aI(this)};G().prototype.atan2=function(n){return this.throwIfDisposed(),cI(this,n)};G().prototype.atanh=function(){return this.throwIfDisposed(),hI(this)};G().prototype.avgPool=function(n,t,e,s){return this.throwIfDisposed(),id(this,n,t,e,s)};G().prototype.batchToSpaceND=function(n,t){return this.throwIfDisposed(),ad(this,n,t)};G().prototype.batchNorm=function(n,t,e,s,o){return this.throwIfDisposed(),$c(this,n,t,e,s,o)};G().prototype.broadcastTo=function(n){return this.throwIfDisposed(),Er(this,n)};G().prototype.cast=function(n){return this.throwIfDisposed(),st(this,n)};G().prototype.ceil=function(){return this.throwIfDisposed(),PI(this)};G().prototype.clipByValue=function(n,t){return this.throwIfDisposed(),on(this,n,t)};G().prototype.concat=function(n,t){return this.throwIfDisposed(),n instanceof Lt&&(n=[n]),ke([this,...n],t)};G().prototype.conv1d=function(n,t,e,s,o,r){return this.throwIfDisposed(),Pg(this,n,t,e,s,o,r)};G().prototype.conv2dTranspose=function(n,t,e,s,o){return this.throwIfDisposed(),zg(this,n,t,e,s,o)};G().prototype.conv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),ho(this,n,t,e,s,o,r)};G().prototype.cos=function(){return this.throwIfDisposed(),cd(this)};G().prototype.cosh=function(){return this.throwIfDisposed(),Vg(this)};G().prototype.cumprod=function(n,t,e){return this.throwIfDisposed(),Tu(this,n,t,e)};G().prototype.cumsum=function(n,t,e){return this.throwIfDisposed(),Wg(this,n,t,e)};G().prototype.depthToSpace=function(n,t){return this.throwIfDisposed(),cv(this,n,t)};G().prototype.depthwiseConv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),ud(this,n,t,e,s,o,r)};G().prototype.dilation2d=function(n,t,e,s,o){return this.throwIfDisposed(),dv(this,n,t,e,s,o)};G().prototype.divNoNan=function(n){return this.throwIfDisposed(),xv(this,n)};G().prototype.div=function(n){return this.throwIfDisposed(),ct(this,n)};G().prototype.dot=function(n){return this.throwIfDisposed(),yv(this,n)};G().prototype.elu=function(){return this.throwIfDisposed(),Cc(this)};G().prototype.equal=function(n){return this.throwIfDisposed(),fn(this,n)};G().prototype.erf=function(){return this.throwIfDisposed(),Ug(this)};G().prototype.euclideanNorm=function(n,t){return this.throwIfDisposed(),Av(this,n,t)};G().prototype.exp=function(){return this.throwIfDisposed(),rn(this)};G().prototype.expandDims=function(n){return this.throwIfDisposed(),Me(this,n)};G().prototype.expm1=function(){return this.throwIfDisposed(),Ov(this)};G().prototype.fft=function(){return this.throwIfDisposed(),Nd(this)};G().prototype.flatten=function(){return this.throwIfDisposed(),L(this,[this.size])};G().prototype.floor=function(){return this.throwIfDisposed(),vc(this)};G().prototype.floorDiv=function(n){return this.throwIfDisposed(),Lg(this,n)};G().prototype.gather=function(n,t,e){return this.throwIfDisposed(),dd(this,n,t,e)};G().prototype.greaterEqual=function(n){return this.throwIfDisposed(),So(this,n)};G().prototype.greater=function(n){return this.throwIfDisposed(),Ze(this,n)};G().prototype.ifft=function(){return this.throwIfDisposed(),Xa(this)};G().prototype.irfft=function(){return this.throwIfDisposed(),ox(this)};G().prototype.isFinite=function(){return this.throwIfDisposed(),Gv(this)};G().prototype.isInf=function(){return this.throwIfDisposed(),qv(this)};G().prototype.isNaN=function(){return this.throwIfDisposed(),Kv(this)};G().prototype.leakyRelu=function(n){return this.throwIfDisposed(),fd(this,n)};G().prototype.lessEqual=function(n){return this.throwIfDisposed(),nr(this,n)};G().prototype.less=function(n){return this.throwIfDisposed(),qa(this,n)};G().prototype.localResponseNormalization=function(n,t,e,s){return this.throwIfDisposed(),Qv(this,n,t,e,s)};G().prototype.logSigmoid=function(){return this.throwIfDisposed(),ik(this)};G().prototype.logSoftmax=function(n){return this.throwIfDisposed(),jg(this,n)};G().prototype.logSumExp=function(n,t){return this.throwIfDisposed(),gd(this,n,t)};G().prototype.log=function(){return this.throwIfDisposed(),mn(this)};G().prototype.log1p=function(){return this.throwIfDisposed(),md(this)};G().prototype.logicalAnd=function(n){return this.throwIfDisposed(),Gn(this,n)};G().prototype.logicalNot=function(){return this.throwIfDisposed(),xd(this)};G().prototype.logicalOr=function(n){return this.throwIfDisposed(),Kg(this,n)};G().prototype.logicalXor=function(n){return this.throwIfDisposed(),fk(this,n)};G().prototype.matMul=function(n,t,e){return this.throwIfDisposed(),Et(this,n,t,e)};G().prototype.maxPool=function(n,t,e,s){return this.throwIfDisposed(),bd(this,n,t,e,s)};G().prototype.max=function(n,t){return this.throwIfDisposed(),Nn(this,n,t)};G().prototype.maximum=function(n){return this.throwIfDisposed(),zs(this,n)};G().prototype.mean=function(n,t){return this.throwIfDisposed(),te(this,n,t)};G().prototype.min=function(n,t){return this.throwIfDisposed(),Ha(this,n,t)};G().prototype.minimum=function(n){return this.throwIfDisposed(),jo(this,n)};G().prototype.mirrorPad=function(n,t){return this.throwIfDisposed(),Ck(this,n,t)};G().prototype.mod=function(n){return this.throwIfDisposed(),vk(this,n)};G().prototype.mul=function(n){return this.throwIfDisposed(),A(this,n)};G().prototype.neg=function(){return this.throwIfDisposed(),Ht(this)};G().prototype.norm=function(n,t,e){return this.throwIfDisposed(),Ic(this,n,t,e)};G().prototype.notEqual=function(n){return this.throwIfDisposed(),Lr(this,n)};G().prototype.oneHot=function(n,t=1,e=0){return this.throwIfDisposed(),Xg(this,n,t,e)};G().prototype.onesLike=function(){return this.throwIfDisposed(),gn(this)};G().prototype.pad=function(n,t){return this.throwIfDisposed(),wd(this,n,t)};G().prototype.pool=function(n,t,e,s,o,r){return this.throwIfDisposed(),_k(this,n,t,e,s,o,r)};G().prototype.pow=function(n){return this.throwIfDisposed(),fo(this,n)};G().prototype.prelu=function(n){return this.throwIfDisposed(),Cd(this,n)};G().prototype.prod=function(n,t){return this.throwIfDisposed(),Mk(this,n,t)};G().prototype.reciprocal=function(){return this.throwIfDisposed(),cS(this)};G().prototype.relu=function(){return this.throwIfDisposed(),ds(this)};G().prototype.relu6=function(){return this.throwIfDisposed(),Yg(this)};G().prototype.reshapeAs=function(n){return this.throwIfDisposed(),L(this,n.shape)};G().prototype.reshape=function(n){return this.throwIfDisposed(),L(this,n)};G().prototype.resizeBilinear=function(n,t,e){return this.throwIfDisposed(),px(this,n,t,e)};G().prototype.resizeNearestNeighbor=function(n,t,e){return this.throwIfDisposed(),fx(this,n,t,e)};G().prototype.reverse=function(n){return this.throwIfDisposed(),go(this,n)};G().prototype.rfft=function(){return this.throwIfDisposed(),Td(this)};G().prototype.round=function(){return this.throwIfDisposed(),Zg(this)};G().prototype.rsqrt=function(){return this.throwIfDisposed(),Jg(this)};G().prototype.selu=function(){return this.throwIfDisposed(),Qg(this)};G().prototype.separableConv2d=function(n,t,e,s,o,r){return this.throwIfDisposed(),tx(this,n,t,e,s,o,r)};G().prototype.sigmoid=function(){return this.throwIfDisposed(),er(this)};G().prototype.sign=function(){return this.throwIfDisposed(),bS(this)};G().prototype.sin=function(){return this.throwIfDisposed(),ex(this)};G().prototype.sinh=function(){return this.throwIfDisposed(),nx(this)};G().prototype.slice=function(n,t){return this.throwIfDisposed(),_t(this,n,t)};G().prototype.softmax=function(n){return this.throwIfDisposed(),Sd(this,n)};G().prototype.softplus=function(){return this.throwIfDisposed(),ji(this)};G().prototype.spaceToBatchND=function(n,t){return this.throwIfDisposed(),$d(this,n,t)};G().prototype.split=function(n,t){return this.throwIfDisposed(),sn(this,n,t)};G().prototype.sqrt=function(){return this.throwIfDisposed(),Ne(this)};G().prototype.square=function(){return this.throwIfDisposed(),Vt(this)};G().prototype.squaredDifference=function(n){return this.throwIfDisposed(),rx(this,n)};G().prototype.squeeze=function(n){return this.throwIfDisposed(),Xi(this,n)};G().prototype.stack=function(n,t){this.throwIfDisposed();const e=n instanceof Lt?[this,n]:[this,...n];return Hn(e,t)};G().prototype.step=function(n){return this.throwIfDisposed(),Yi(this,n)};G().prototype.stridedSlice=function(n,t,e,s,o,r,i,a){return this.throwIfDisposed(),LS(this,n,t,e,s,o,r,i,a)};G().prototype.sub=function(n){return this.throwIfDisposed(),rt(this,n)};G().prototype.sum=function(n,t){return this.throwIfDisposed(),it(this,n,t)};G().prototype.tan=function(){return this.throwIfDisposed(),PS(this)};G().prototype.tanh=function(){return this.throwIfDisposed(),wc(this)};G().prototype.tile=function(n){return this.throwIfDisposed(),kn(this,n)};G().prototype.toBool=function(){return this.throwIfDisposed(),st(this,"bool")};G().prototype.toFloat=function(){return this.throwIfDisposed(),st(this,"float32")};G().prototype.toInt=function(){return this.throwIfDisposed(),st(this,"int32")};G().prototype.topk=function(n,t){return this.throwIfDisposed(),BS(this,n,t)};G().prototype.transpose=function(n){return this.throwIfDisposed(),vt(this,n)};G().prototype.unique=function(n){return this.throwIfDisposed(),US(this,n)};G().prototype.unsortedSegmentSum=function(n,t){return this.throwIfDisposed(),lx(this,n,t)};G().prototype.unstack=function(n){return this.throwIfDisposed(),xo(this,n)};G().prototype.where=function(n,t){return this.throwIfDisposed(),Se(n,this,t)};G().prototype.zerosLike=function(){return this.throwIfDisposed(),kt(this)};class Ln extends Error{constructor(t){super(t),Object.setPrototypeOf(this,Ln.prototype)}}class dn extends Error{constructor(t){super(t),Object.setPrototypeOf(this,dn.prototype)}}class F extends Error{constructor(t){super(t),Object.setPrototypeOf(this,F.prototype)}}class yt extends Error{constructor(t){super(t),Object.setPrototypeOf(this,yt.prototype)}}class hp extends Error{constructor(t){super(t),Object.setPrototypeOf(this,hp.prototype)}}class rb{constructor(t){this.maxEntries=t||100,this.cache=new Map}get(t){let e;return this.cache.has(t)&&(e=this.cache.get(t),this.cache.delete(t),this.cache.set(t,e)),e}put(t,e){if(this.cache.has(t))this.cache.delete(t);else if(this.cache.size>=this.maxEntries){const s=this.cache.keys().next().value;this.cache.delete(s)}this.cache.set(t,e)}getMaxEntries(){return this.maxEntries}setMaxEntries(t){if(t<0)throw new Error(`The maxEntries of LRU caches must be at least 0, but got ${t}.`);if(this.maxEntries>t)for(let e=0;e<this.maxEntries-t;e++){const s=this.cache.keys().next().value;this.cache.delete(s)}this.maxEntries=t}}function bo(n,t){if(Array.isArray(n)){let e=[];for(let s=0;s<t;s++)e=e.concat(n);return e}else{const e=new Array(t);return e.fill(n),e}}function Mn(n,t){if(!n)throw new hp(t)}function Uf(n,t){let e=0;for(const s of n)s===t&&e++;return e}function Pe(n){return n.length===1?n[0]:n}function At(n){return Array.isArray(n)?n:[n]}function Zn(n){const e=n.replace(/(.)([A-Z][a-z0-9]+)/g,"$1_$2").replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase();return e[0]!=="_"?e:"private"+e}function Ys(n){return n.length<=1||n.indexOf("_")===-1?n:n.replace(/[_]+(\w|$)/g,(t,e)=>e.toUpperCase())}let ln={};function dp(n){if(n==null)return null;const t={};return t.className=n.getClassName(),t.config=n.getConfig(),t}function Du(n){if(!(n==null||typeof n!="object"))if(Array.isArray(n))n.forEach(t=>Du(t));else{const t=Object.keys(n);for(const e of t){const s=n[e];s!=null&&typeof s=="object"&&(!Array.isArray(s)&&s.type==="ndarray"&&typeof s.value=="number"?n[e]=s.value:Du(s))}}}function ta(n,t={},e={},s="object",o=!1){if(typeof n=="string"){const r=n;let i;if(r in e)i=e[r];else if(r in ln)i=ln[r];else if(i=t[r],i==null)throw new F(`Unknown ${s}: ${n}. This may be due to one of the following reasons:
1. The ${s} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${s} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return i}else{const r=n;if(r.className==null||r.config==null)throw new F(`${s}: Improper config format: ${JSON.stringify(r)}.
'className' and 'config' must set.`);const i=r.className;let a,l;if(i in e?[a,l]=e[i]:i in ln?[a,l]=ln.className:i in t&&([a,l]=t[i]),a==null)throw new F(`Unknown ${s}: ${i}. This may be due to one of the following reasons:
1. The ${s} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${s} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(l!=null){const c={};for(const p of Object.keys(ln))c[p]=ln[p];for(const p of Object.keys(e))c[p]=e[p];const u=r.config;u.customObjects=c;const h=Object.assign({},ln);for(const p of Object.keys(e))ln[p]=e[p];Du(r.config);const d=l(a,r.config,e,o);return ln=Object.assign({},h),d}else{const c=Object.assign({},ln);for(const h of Object.keys(e))ln[h]=e[h];const u=new a(r.config);return ln=Object.assign({},c),u}}}function lR(n,t){return n<t?-1:n>t?1:0}function ma(n,t){return-1*lR(n,t)}function ks(n){if(n==null)return n;const t=[];for(const e of n)t.indexOf(e)===-1&&t.push(e);return t}function cR(n){if(n==null)throw new F(`Invalid value in obj: ${JSON.stringify(n)}`);for(const t in n)if(n.hasOwnProperty(t))return!1;return!0}function To(n,t,e){if(e!=null&&n.indexOf(e)<0)throw new F(`${e} is not a valid ${t}.  Valid values are ${n} or null/undefined.`)}function pp(n,t,e=0,s=1/0){return Mn(e>=0),Mn(s>=e),Array.isArray(n)&&n.length>=e&&n.length<=s&&n.every(o=>typeof o===t)}function fe(n,t){Array.isArray(n)?(I(n.length>0,()=>`${t} is unexpectedly an empty array.`),n.forEach((e,s)=>fe(e,`element ${s+1} of ${t}`))):I(Number.isInteger(n)&&n>0,()=>`Expected ${t} to be a positive integer, but got ${ib(n)}.`)}function ib(n){return n===null?"null":Array.isArray(n)?"["+n.map(t=>ib(t)).join(",")+"]":typeof n=="string"?`"${n}"`:`${n}`}function uR(n,t,e){let s=e!=null?e():xe(),o;return(...i)=>{const a=e!=null?e():xe();return a-s<t||(s=a,o=n(...i)),o}}function ab(n){return n==="relu"?"relu":n==="linear"?"linear":n==="elu"?"elu":null}let hR=0;function lb(){return hR++}const ga={};function Ac(n=""){return n in ga||(ga[n]=0),ga[n]+=1,n+ga[n].toString()}const dR=["channelsFirst","channelsLast"],pR=["nearest","bilinear"],fR=["valid","same","causal"],mR=["max","avg"],gR=["sum","mul","concat","ave"];const Oo=new Map;function se(n){To(dR,"DataFormat",n)}function xR(n){To(pR,"InterpolationFormat",n)}function an(n){To(fR,"PaddingMode",n)}function cb(n){To(mR,"PoolMode",n)}const Rr=[],Gf="/";function no(n,t){Rr.push(n);try{const e=t();return Rr.pop(),e}catch(e){throw Rr.pop(),e}}function bR(){return Rr.length===0?"":Rr.join(Gf)+Gf}function ub(n){if(!db(n))throw new Error("Not a valid tensor name: '"+n+"'");return bR()+n}function hb(n){if(!db(n))throw new Error("Not a valid tensor name: '"+n+"'");Oo.has(n)||Oo.set(n,0);const t=Oo.get(n);if(Oo.set(n,Oo.get(n)+1),t>0){const e=`${n}_${t}`;return Oo.set(e,1),e}else return n}const yR=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function db(n){return!!n.match(yR)}function wR(n){return n===parseInt(n.toString(),10)}function Ss(n,t,e){t==null&&(t=0),e==null&&(e=n.length);let s=1;for(let o=t;o<e;++o)s*=n[o];return s}function Ko(n){if(n.length===0)return Number.NaN;let t=Number.POSITIVE_INFINITY;for(let e=0;e<n.length;e++){const s=n[e];s<t&&(t=s)}return t}function As(n){if(n.length===0)return Number.NaN;let t=Number.NEGATIVE_INFINITY;for(let e=0;e<n.length;e++){const s=n[e];s>t&&(t=s)}return t}function Rn(n,t){if(t<n)throw new F(`end (${t}) < begin (${n}) is forbidden.`);const e=[];for(let s=n;s<t;++s)e.push(s);return e}let ru;function ie(){return ru==null&&(ru=Sg().epsilon()),ru}function An(){return"channelsLast"}function Wn(n,t){return st(n,t)}function ea(n,t=-1){const e=n.shape.slice();return t<0&&(t=e.length+t+1),e.splice(t,0,1),L(n,e)}function $R(n,t){return z(()=>{if(n.shape.length!==2)throw new F(`repeat() expects a rank-2 tensor, but received a rank-${n.shape.length} tensor.`);const e=ea(n,1);return Fu(e,[1,t,1])})}function CR(n){const t=[Ss(n.shape)];return L(n,t)}function IR(n){if(n.rank<=1)throw new F(`batchFlatten requires a minimum rank of 2. Got rank: ${n.rank}.`);const t=[n.shape[0],Ss(n.shape,1)];return L(n,t)}function so(n,t,e){return z(()=>{switch(n.rank){case 1:return vd(n,t,e);case 2:return sx(n,[t,0],[e,n.shape[1]]);case 3:return kd(n,[t,0,0],[e,n.shape[1],n.shape[2]]);case 4:return Ka(n,[t,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3]]);case 5:return _t(n,[t,0,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3],n.shape[4]]);case 6:return _t(n,[t,0,0,0,0,0],[e,n.shape[1],n.shape[2],n.shape[3],n.shape[4],n.shape[5]]);default:throw new F(`sliceAlongFirstAxis() received an unsupported tensor rank: ${n.rank}`)}})}function iu(n,t,e){return z(()=>{switch(n.rank){case 1:return vd(n,t,e);case 2:return sx(n,[0,t],[n.shape[0],e]);case 3:return kd(n,[0,0,t],[n.shape[0],n.shape[1],e]);case 4:return Ka(n,[0,0,0,t],[n.shape[0],n.shape[1],n.shape[2],e]);default:throw new F(`sliceAlongLastAxis() received an unsupported tensor rank: ${n.rank}`)}})}function xa(n,t,e,s){return z(()=>{switch(n.rank){case 1:return vd(n,t,e);case 2:switch(s){case 1:return so(n,t,e);case 2:return iu(n,t,e);default:throw new F(`The axis is not within the rank of the tensor ${s}`)}case 3:switch(s){case 1:return so(n,t,e);case 2:return kd(n,[0,t,0],[n.shape[0],e,n.shape[2]]);case 3:return iu(n,t,e);default:throw new F(`The axis is not within the rank of the tensor ${s}`)}case 4:switch(s){case 1:return so(n,t,e);case 2:return Ka(n,[0,t,0,0],[n.shape[0],e,n.shape[2],n.shape[3]]);case 3:return Ka(n,[0,0,t,0],[n.shape[0],n.shape[1],e,n.shape[3]]);case 4:return iu(n,t,e);default:throw new F(`The axis is not within the rank of the tensor ${s}`)}default:throw new F(`sliceAlongLastAxis() received an unsupported tensor rank: ${n.rank}`)}})}function fp(n,t=-1){let e;return t<0&&(e=n[0].rank,e!==0?t=e:t=0),t===n[0].rank&&(t=-1),ke(n,t)}function Hf(n,t){switch(n.rank){case 1:return VI([n,t]);case 2:return UI([n,t],0);case 3:return HI([n,t],0);case 4:return jI([n,t],0);default:throw new F(`concatAlongFirstAxis() received an unsupported tensor rank: ${n.rank}`)}}function Fu(n,t){if(Array.isArray(t)||(t=[t]),n.rank!==t.length)throw new F(`The length of input n (${t.length}) does not match the number of dimensions in input x (${n.rank})`);return kn(n,t)}function Dc(n,t=0,e=1,s,o){return rS(n,t,e,s,o)}function Un(n,t,e,s){if(n.rank<2||t.rank<2)throw new yt(`dot requires both inputs to be rank >= 2 but got x shape = ${n.shape} and y shape = ${t.shape}`);if(t.rank>=3){const o=n.shape.slice(-1)[0],r=t.shape.slice(-2)[0];if(o!==r)throw new yt(`If rank y >= 3, then the second last dim of y must equal the last dim of x but got x shape = ${n.shape} and  y shape = ${t.shape}`)}if(n.rank===2&&t.rank===2)return Ef({a:n,b:t,transposeA:!1,transposeB:!1,bias:s?_u(n.rank,s,An()):null,activation:e});{const o=n.shape.slice(),r=o.pop();n=L(n,[-1,r]);const i=t.shape.slice(),a=i.pop(),l=i.pop(),c=[...i,a],u=Array.from({length:t.rank},(f,m)=>m===0?t.rank-2:m<=t.rank-2?m-1:m);t=L(vt(t,u),[l,-1]);const h=[...o,...c];return L(Ef({a:n,b:t,transposeA:!1,transposeB:!1,bias:s?_u(n.rank,s,An()):null,activation:e}),h)}}function pb(n,t,e){return z(()=>(Array.isArray(t)?t=Be(t,"int32"):t=st(t,"int32"),dd(n,t,e)))}function na(n){return A(n,n)}function _u(n,t,e){const s=t.shape;if(t.rank!==1&&t.rank!==n)throw new F(`Unexpected bias dimensions: ${t.rank}; expected it to be 1 or ${n}`);if(n===5){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1,1,1]):L(t,[1,s[3],s[0],s[1],s[2]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,1,1,s[0]]):L(t,[1].concat(s))}else if(n===4){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1,1]):L(t,[1,s[2],s[0],s[1]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,1,s[0]]):L(t,[1].concat(s))}else if(n===3){if(e==="channelsFirst")return s.length===1?L(t,[1,s[0],1]):L(t,[1,s[1],s[0]]);if(e==="channelsLast")return s.length===1?L(t,[1,1,s[0]]):L(t,[1].concat(s))}else if(n<3)return t;throw new F(`Unsupported input rank by biasAdd: ${t.rank}`)}function _n(n,t,e){return z(()=>(e==null&&(e=An()),se(e),J(n,_u(n.rank,t,e))))}function vR(n,t=1){if(t!==1)throw new yt(`Support for alpha values other than 1 (${t}) is not implemented yet.`);return Cc(n)}function kR(n){return z(()=>ct(n,J(he(n),1)))}function fb(n,t,e,s){return z(()=>ZS(n,t,e,s))}function SR(n){return z(()=>{const t=J(.5,A(.2,n));return on(t,0,1)})}function sa(n,t,e=!1){return e?n():t()}const NR=["fanIn","fanOut","fanAvg"],TR=["normal","uniform","truncatedNormal"];function ER(n){To(NR,"FanMode",n)}function RR(n){To(TR,"Distribution",n)}class bn extends No{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}}class mb extends bn{apply(t,e){return pe(t,e)}}mb.className="Zeros";X(mb);class mp extends bn{apply(t,e){return hs(t,e)}}mp.className="Ones";X(mp);class gb extends bn{constructor(t){if(super(),typeof t!="object")throw new F(`Expected argument of type ConstantConfig but got ${t}`);if(t.value===void 0)throw new F(`config must have value set but got ${t}`);this.value=t.value}apply(t,e){return z(()=>A(mt(this.value),hs(t,e)))}getConfig(){return{value:this.value}}}gb.className="Constant";X(gb);class xb extends bn{constructor(t){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=t.minval||this.DEFAULT_MINVAL,this.maxval=t.maxval||this.DEFAULT_MAXVAL,this.seed=t.seed}apply(t,e){return Ki(t,this.minval,this.maxval,e,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}}xb.className="RandomUniform";X(xb);class bb extends bn{constructor(t){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=t.mean||this.DEFAULT_MEAN,this.stddev=t.stddev||this.DEFAULT_STDDEV,this.seed=t.seed}apply(t,e){if(e=e||"float32",e!=="float32"&&e!=="int32")throw new yt(`randomNormal does not support dType ${e}.`);return Dc(t,this.mean,this.stddev,e,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}bb.className="RandomNormal";X(bb);class yb extends bn{constructor(t){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=t.mean||this.DEFAULT_MEAN,this.stddev=t.stddev||this.DEFAULT_STDDEV,this.seed=t.seed}apply(t,e){if(e=e||"float32",e!=="float32"&&e!=="int32")throw new yt(`truncatedNormal does not support dType ${e}.`);return ax(t,this.mean,this.stddev,e,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}yb.className="TruncatedNormal";X(yb);class wb extends bn{constructor(t){super(),this.gain=t.gain!=null?t.gain:1}apply(t,e){return z(()=>{if(t.length!==2||t[0]!==t[1])throw new F("Identity matrix initializer can only be used for 2D square matrices.");return A(this.gain,qg(t[0]))})}getConfig(){return{gain:this.gain}}}wb.className="Identity";X(wb);function AR(n,t="channelsLast"){let e,s;if(se(t),n.length===2)e=n[0],s=n[1];else if([3,4,5].indexOf(n.length)!==-1){if(t==="channelsFirst"){const o=Ss(n,2);e=n[1]*o,s=n[0]*o}else if(t==="channelsLast"){const o=Ss(n,0,n.length-2);e=n[n.length-2]*o,s=n[n.length-1]*o}}else{const o=Ss(n);e=Math.sqrt(o),s=Math.sqrt(o)}return[e,s]}class Xe extends bn{constructor(t){if(super(),t.scale<0)throw new F(`scale must be a positive float. Got: ${t.scale}`);this.scale=t.scale==null?1:t.scale,this.mode=t.mode==null?"fanIn":t.mode,ER(this.mode),this.distribution=t.distribution==null?"normal":t.distribution,RR(this.distribution),this.seed=t.seed}apply(t,e){const s=AR(t),o=s[0],r=s[1];let i=this.scale;if(this.mode==="fanIn"?i/=Math.max(1,o):this.mode==="fanOut"?i/=Math.max(1,r):i/=Math.max(1,(o+r)/2),this.distribution==="normal"){const a=Math.sqrt(i);if(e=e||"float32",e!=="float32"&&e!=="int32")throw new yt(`${this.getClassName()} does not support dType ${e}.`);return ax(t,0,a,e,this.seed)}else{const a=Math.sqrt(3*i);return Ki(t,-a,a,e,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}}Xe.className="VarianceScaling";X(Xe);class gp extends Xe{constructor(t){super({scale:1,mode:"fanAvg",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}gp.className="GlorotUniform";X(gp);class xp extends Xe{constructor(t){super({scale:1,mode:"fanAvg",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}xp.className="GlorotNormal";X(xp);class bp extends Xe{constructor(t){super({scale:2,mode:"fanIn",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}bp.className="HeNormal";X(bp);class yp extends Xe{constructor(t){super({scale:2,mode:"fanIn",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}yp.className="HeUniform";X(yp);class wp extends Xe{constructor(t){super({scale:1,mode:"fanIn",distribution:"normal",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}wp.className="LeCunNormal";X(wp);class $p extends Xe{constructor(t){super({scale:1,mode:"fanIn",distribution:"uniform",seed:t==null?null:t.seed})}getClassName(){return Xe.className}}$p.className="LeCunUniform";X($p);class $b extends bn{constructor(t){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=t.gain==null?this.DEFAULT_GAIN:t.gain,this.seed=t.seed}apply(t,e){return z(()=>{if(t.length<2)throw new yt("Shape must be at least 2D.");if(e!=="int32"&&e!=="float32"&&e!==void 0)throw new TypeError(`Unsupported data type ${e}.`);e=e;const s=U(t.slice(0,-1)),o=t[t.length-1],r=s*o;r>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${r}) elements: Slowness may result.`);const i=[Math.max(o,s),Math.min(o,s)],a=Dc(i,0,1,e,this.seed),l=ON.qr(a,!1);let c=l[0];const h=l[1].flatten().stridedSlice([0],[Math.min(o,s)*Math.min(o,s)],[Math.min(o,s)+1]);return c=A(c,h.sign()),s<o&&(c=c.transpose()),A(mt(this.gain),c.reshape(t))})}getConfig(){return{gain:this.gain,seed:this.seed}}}$b.className="Orthogonal";X($b);const qf={constant:"Constant",glorotNormal:"GlorotNormal",glorotUniform:"GlorotUniform",heNormal:"HeNormal",heUniform:"HeUniform",identity:"Identity",leCunNormal:"LeCunNormal",leCunUniform:"LeCunUniform",ones:"Ones",orthogonal:"Orthogonal",randomNormal:"RandomNormal",randomUniform:"RandomUniform",truncatedNormal:"TruncatedNormal",varianceScaling:"VarianceScaling",zeros:"Zeros"};function jf(n,t={}){return ta(n,tn.getMap().classNameMap,t,"initializer")}function Kt(n){return dp(n)}function Gt(n){if(typeof n=="string"){const t=n in qf?qf[n]:n;if(t==="GlorotNormal")return new xp;if(t==="GlorotUniform")return new gp;if(t==="HeNormal")return new bp;if(t==="HeUniform")return new yp;if(t==="LeCunNormal")return new wp;if(t==="LeCunUniform")return new $p;{const e={};return e.className=t,e.config={},jf(e)}}else return n instanceof bn?n:jf(n)}function Ou(n){return Array.isArray(n)&&Array.isArray(n[0])}function Ya(n){return n.length===0?[]:Array.isArray(n[0])?n:[n]}function ft(n){let t;if(Array.isArray(n)){if(n.length!==1)throw new F(`Expected Tensor length to be 1; got ${n.length}`);t=n[0]}else t=n;return t}function Nt(n){if(Array.isArray(n)&&Array.isArray(n[0])){if(n.length===1)return n=n,n[0];throw new F(`Expected exactly 1 Shape; got ${n.length}`)}else return n}function Za(n){let t=0;for(const e of n)e.shape.length===0?t+=1:t+=e.shape.reduce((s,o)=>s*o);return t}const Kf="Variable";class DR{constructor(t,e="float32",s=Kf,o=!0,r=null){this.dtype=e??"float32",this.shape=t.shape,this.id=lb(),s=s??Kf,this.originalName=ub(s),this.name=hb(this.originalName),this.trainable_=o,this.constraint=r,this.val=qS(t,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(t){return this.assertNotDisposed(),FR(this.val,t),this.val.id!==t.id&&(this.val.assign(t),this.constraint!=null&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw new Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(t){this.trainable_=t,this.val.trainable=t}}function FR(n,t){if(n.shape.toString()!==t.shape.toString())throw new Error("Shape mismatch: "+JSON.stringify(n.shape)+" vs. "+JSON.stringify(t.shape))}function Lu(n){return n.map(t=>t.read())}function Cp(n){n.forEach(t=>{t[0].write(t[1])})}class ae{constructor(t){this.dtype=t.dtype,this.shape=t.shape,t.shape!=null?this.ndim=t.shape.length:this.ndim=t.ndim,this.maxNDim=t.maxNDim,this.minNDim=t.minNDim,this.axes=t.axes||{}}}class qn{constructor(t,e,s,o,r,i,a){this.dtype=t,this.shape=e,this.sourceLayer=s,this.inputs=o,this.callArgs=r,this.outputTensorIndex=a,this.id=lb(),i!=null&&(this.originalName=ub(i),this.name=hb(this.originalName)),this.rank=e.length}}let _R=0;class Fc{constructor(t,e){this.callArgs=e,this.id=_R++,this.outboundLayer=t.outboundLayer,this.inboundLayers=t.inboundLayers,this.nodeIndices=t.nodeIndices,this.tensorIndices=t.tensorIndices,this.inputTensors=t.inputTensors,this.outputTensors=t.outputTensors,this.inputMasks=t.inputMasks,this.outputMasks=t.outputMasks,this.inputShapes=t.inputShapes,this.outputShapes=t.outputShapes;for(const s of t.inboundLayers)s?.outboundNodes.push(this);t.outboundLayer.inboundNodes.push(this)}getConfig(){const t=[];for(const e of this.inboundLayers)e!=null?t.push(e.name):t.push(null);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:t,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}}let OR=0;class It extends No{constructor(t={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=OR++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let e=t.name;if(!e){const s=this.getClassName();e=Zn(s)+"_"+Ac(s)}if(this.name=e,this.trainable_=t.trainable==null?!0:t.trainable,t.inputShape!=null||t.batchInputShape!=null){let s;if(t.batchInputShape!=null)s=t.batchInputShape;else if(t.inputShape!=null){let r=null;t.batchSize!=null&&(r=t.batchSize),s=[r].concat(t.inputShape)}this.batchInputShape=s;let o=t.dtype;o==null&&(o=t.inputDType),o==null&&(o="float32"),this.dtype=o}t.weights!=null?this.initialWeights=t.weights:this.initialWeights=null,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(t,e){return t.name+"_ib-"+e.toString()}getNodeAtIndex(t,e){if(this.inboundNodes.length===0)throw new dn(`The layer has never been called and thus has no defined ${e}.`);if(this.inboundNodes.length<=t)throw new F(`Asked to get ${e} at node ${t}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[t]}getInputAt(t){return Pe(this.getNodeAtIndex(t,"input").inputTensors)}getOutputAt(t){return Pe(this.getNodeAtIndex(t,"output").outputTensors)}get input(){if(this.inboundNodes.length>1)throw new Ln(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(this.inboundNodes.length===0)throw new Ln(`Layer ${this.name} is not connected, no input to return.`);return Pe(this.getNodeAtIndex(0,"input").inputTensors)}get output(){if(this.inboundNodes.length===0)throw new Ln(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new Ln(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return Pe(this.getNodeAtIndex(0,"output").outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(t=>t())}get updates(){return this._updates}get built(){return this._built}set built(t){this._built=t}get trainable(){return this.trainable_}set trainable(t){this._trainableWeights.forEach(e=>e.trainable=t),this.trainable_=t}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(t=>t.trainable):[]}set trainableWeights(t){this._trainableWeights=t}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(t=>!t.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(t){this._nonTrainableWeights=t}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw new Error("Cannot call the resetStates() method of a non-stateful Layer object.")}assertInputCompatibility(t){const e=At(t);if(this.inputSpec==null||this.inputSpec.length===0)return;const s=At(this.inputSpec);if(e.length!==s.length)throw new F(`Layer ${this.name} expects ${s.length} inputs, but it received ${e.length} input tensors. Input received: ${t}`);for(let o=0;o<e.length;o++){const r=e[o],i=s[o];if(i==null)continue;const a=r.rank;if(i.ndim!=null&&a!==i.ndim)throw new F(`Input ${o} is incompatible with layer ${this.name}: expected ndim=${i.ndim}, found ndim=${a}`);if(i.maxNDim!=null&&a>i.maxNDim)throw new F(`Input ${o} is incompatible with layer ${this.name}: expected max_ndim=${i.maxNDim}, found ndim=${a}`);if(i.minNDim!=null&&a<i.minNDim)throw new F(`Input ${o} is incompatible with layer ${this.name}: expected min_ndim=${i.minNDim}, found ndim=${a}.`);if(i.dtype!=null&&r.dtype!==i.dtype)throw new F(`Input ${o} is incompatible with layer ${this.name} : expected dtype=${i.dtype}, found dtype=${r.dtype}.`);if(i.axes){const l=r.shape;for(const c in i.axes){const u=Number(c),h=i.axes[c],d=u>=0?l[u]:l[l.length+u];if(h!=null&&[h,null].indexOf(d)===-1)throw new F(`Input ${o} is incompatible with layer ${this.name}: expected axis ${u} of input shape to have value ${h} but got shape ${l}.`)}}if(i.shape!=null)for(let l=0;l<i.shape.length;++l){const c=i.shape[l],u=r.shape[l];if(c!=null&&u!=null&&c!==u)throw new F(`Input ${o} is incompatible with layer ${this.name}: expected shape=${i.shape}, found shape=${r.shape}.`)}}}call(t,e){return t}invokeCallHook(t,e){this._callHook!=null&&this._callHook(t,e)}setCallHook(t){this._callHook=t}clearCallHook(){this._callHook=null}apply(t,e){e=e||{},this.assertNotDisposed();const s=At(t),o=PR(t),r=zR(t);if(o===r)throw new F("Arguments to apply() must be all SymbolicTensors or all Tensors");return no(this.name,()=>{if(!this.built){this.assertInputCompatibility(t);const i=[];for(const a of At(t))i.push(a.shape);this.build(Pe(i)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),this._refCount===null&&r&&(this._refCount=1)}if(this.assertInputCompatibility(t),r){let i=this.call(t,e);this.supportsMasking&&this.setMaskMetadata(t,i);const a=At(i),l=[];for(let c of a)s.indexOf(c)!==-1&&(c=c.clone()),l.push(c);if(i=Pe(l),this.activityRegularizer!=null)throw new yt("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return i}else{const i=LR(t),a=this.computeOutputShape(i);let l;const c=MR(t);if(this.warnOnIncompatibleInputShape(Array.isArray(t)?i[0]:i),a!=null&&a.length>0&&Array.isArray(a[0])?l=a.map((u,h)=>new qn(c,u,this,At(t),e,this.name,h)):l=new qn(c,a,this,At(t),e,this.name),this.addInboundNode(t,l,null,null,i,a,e),this._refCount++,this.activityRegularizer!=null)throw new yt("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return l}})}warnOnIncompatibleInputShape(t){if(this.batchInputShape!=null)if(t.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(t)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let e=!1;this.batchInputShape.forEach((s,o)=>{s!=null&&t[o]!=null&&t[o]!==s&&(e=!0)}),e&&console.warn(`The shape of the input tensor (${JSON.stringify(t)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}get outputShape(){if(this.inboundNodes==null||this.inboundNodes.length===0)throw new Ln(`The layer ${this.name} has never been called and thus has no defined output shape.`);const t=[];for(const e of this.inboundNodes){const s=JSON.stringify(e.outputShapes);t.indexOf(s)===-1&&t.push(s)}if(t.length===1){const e=this.inboundNodes[0].outputShapes;return Array.isArray(e)&&Array.isArray(e[0])&&e.length===1?e[0]:e}else throw new Ln(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new dn(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return Za(this.weights)}build(t){this.built=!0}getWeights(t=!1){return Lu(t?this.trainableWeights:this.weights)}setWeights(t){z(()=>{const e=this.weights;if(e.length!==t.length)throw new F(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${t.length}, but the layer was expecting ${e.length} weights. Provided weights: ${t}...`);if(e.length===0)return;const s=[],o=Lu(e);for(let r=0;r<o.length;++r){const i=o[r],a=e[r],l=t[r];if(!Tt(i.shape,l.shape))throw new F(`Layer weight shape ${i.shape} not compatible with provided weight shape ${l.shape}`);s.push([a,l])}Cp(s)})}addWeight(t,e,s,o,r,i,a,l){if(this._addedWeightNames.indexOf(t)!==-1)throw new F(`Duplicate weight name ${t} for layer ${this.name}`);this._addedWeightNames.push(t),s==null&&(s="float32"),this.fastWeightInitDuringBuild&&(o=l!=null?l():Gt("zeros"));const c=o.apply(e,s),u=new DR(c,s,t,i,a);return c.dispose(),r!=null&&this.addLoss(()=>r.apply(u.read())),i==null&&(i=!0),i?this._trainableWeights.push(u):this._nonTrainableWeights.push(u),u}setFastWeightInitDuringBuild(t){this.fastWeightInitDuringBuild=t}addLoss(t){t==null||Array.isArray(t)&&t.length===0||(t=At(t),this._losses!==void 0&&this._losses!==null&&this.losses.push(...t))}computeOutputShape(t){return t}computeMask(t,e){if(!this.supportsMasking){if(e!=null)if(Array.isArray(e))e.forEach(s=>{if(s!=null)throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`);return null}return e}setMaskMetadata(t,e,s){if(!this.supportsMasking)return;const o=this.computeMask(t,s),r=At(e),i=At(o);if(r.length!==i.length)throw new Error(`${this.name} outputs ${r.length} tensors but ${r.length} masks for those tensors`);for(let a=0;a<r.length;a++)r[a].kerasMask=i[a]}addInboundNode(t,e,s,o,r,i,a=null){const l=At(t);e=At(e),s=At(s),o=At(o),r=Ya(r),i=Ya(i);const c=[],u=[],h=[];for(const d of l)c.push(d.sourceLayer),u.push(d.nodeIndex),h.push(d.tensorIndex);new Fc({outboundLayer:this,inboundLayers:c,nodeIndices:u,tensorIndices:h,inputTensors:l,outputTensors:e,inputMasks:s,outputMasks:o,inputShapes:r,outputShapes:i},a);for(let d=0;d<e.length;d++)e[d].sourceLayer=this,e[d].nodeIndex=this.inboundNodes.length-1,e[d].tensorIndex=d}getConfig(){const t={name:this.name,trainable:this.trainable};return this.batchInputShape!=null&&(t.batchInputShape=this.batchInputShape),this.dtype!=null&&(t.dtype=this.dtype),t}disposeWeights(){return this.weights.forEach(t=>t.dispose()),this.weights.length}assertNotDisposed(){if(this._refCount===0)throw new Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw new Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(this._refCount===null)throw new Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let t=0;return--this._refCount===0&&(t=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:t}}}function LR(n){n=At(n);const t=[];for(const e of n)t.push(e.shape);return Pe(t)}function MR(n){return"float32"}function Cb(n,t,e){if((t==null||e!=null&&e>0)&&(t=n.sourceLayer,e=n.nodeIndex),t.inboundNodes.length===0)return[n];{const s=t.inboundNodes[e];if(s.inboundLayers.length===0)return s.inputTensors;{const o=[];for(let r=0;r<s.inboundLayers.length;r++){const i=s.inputTensors[r],a=s.inboundLayers[r],l=s.nodeIndices[r],c=Cb(i,a,l);for(const u of c)o.indexOf(u)===-1&&o.push(u)}return o}}}function PR(n){let t=!0;for(const e of At(n))if(!(e instanceof qn)){t=!1;break}return t}function zR(n){let t=!0;for(const e of At(n))if(e instanceof qn){t=!1;break}return t}class oa extends It{constructor(t){if(super({dtype:t.dtype,name:t.name!=null?t.name:Ac("input").toString()}),t.batchSize==null&&(t.batchSize=null),t.sparse==null&&(t.sparse=!1),this.trainable=!1,this.built=!0,this.sparse=t.sparse,t.inputShape!=null&&t.batchInputShape!=null)throw new F("Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.");let e=t.batchInputShape;if(e==null){if(t.inputShape==null)throw new F("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");e=[t.batchSize].concat(t.inputShape)}else if(t.batchSize!=null)throw new F("Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.");const s=t.dtype||"float32";this.batchInputShape=e,this.dtype=s,this.inputSpec=[{shape:e}];const o=new qn(this.dtype,this.batchInputShape,this,[],{},this.name);o.nodeIndex=0,o.tensorIndex=0,new Fc({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[o],outputTensors:[o],inputMasks:[null],outputMasks:[null],inputShapes:[e],outputShapes:[e]})}apply(t,e){throw new F(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}}oa.className="InputLayer";X(oa);function BR(n){if(n.batchShape==null&&n.shape==null)throw new Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(n.batchShape!=null&&n.shape!=null)throw new F("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let t=n.batchShape;n.shape!=null&&t==null&&(t=[null].concat(n.shape));let e=n.dtype;return e==null&&(e="float32"),new oa({batchInputShape:t,name:n.name,dtype:e,sparse:n.sparse}).inboundNodes[0].outputTensors[0]}function VR(n,t){if(n.dtype==null||n.dtype===t.dtype)return t;try{return st(t,n.dtype)}catch{throw new F(`The dtype of the feed (${t.dtype}) can not be cast to the dtype of the key '${n.name}' (${n.dtype}).`)}}class $s{constructor(t){if(this.id2Value={},this.id2Mask={},this.name2Id={},t instanceof $s)for(const e in t.id2Value)this.id2Value[e]=t.id2Value[e],e in t.id2Mask&&(this.id2Mask[e]=t.id2Mask[e]);else{if(t==null)return;for(const e of t)this.add(e.key,e.value)}}add(t,e,s){if(this.id2Value[t.id]==null)this.id2Value[t.id]=VR(t,e),this.name2Id[t.name]=t.id,s!=null&&(this.id2Mask[t.id]=s);else throw new F(`Duplicate key: name=${t.name}, id=${t.id}`);return this}addFeed(t){this.add(t.key,t.value)}hasKey(t){return this.id2Value[t.id]!=null}names(){return Object.keys(this.name2Id)}getValue(t){if(t instanceof qn){if(this.id2Value[t.id]==null)throw new F(`Nonexistent key: ${t.name}`);return this.id2Value[t.id]}else{const e=this.name2Id[t];if(e==null)throw new F(`Feed dict has no SymbolicTensor name: ${t}`);return this.id2Value[e]}}getMask(t){if(t instanceof qn){if(this.id2Value[t.id]==null)throw new F(`Nonexistent key: ${t.name}`);return this.id2Mask[t.id]}else{const e=this.name2Id[t];if(e==null)throw new F(`Feed dict has no SymbolicTensor name: ${t}`);return this.id2Mask[e]}}disposeMasks(){this.id2Mask!=null&&xt(this.id2Mask)}}const Ja=new rb,Qa=new rb;function WR(n){Ja?.setMaxEntries(n),Qa?.setMaxEntries(n)}function Cr(n,t,e,s){const o=e==null?!1:e.training,r=Array.isArray(n),i=r?n:[n],a=i.map(f=>f.name),l=[],c=t.names();for(const f of a)c.indexOf(f)!==-1?l.push(t.getValue(f)):l.push(null);const u=a.join(",")+"|"+t.names().sort().join(",");let h=Ja.get(u),d;if(h==null){const f=UR(i,t);h=f.sorted,d=f.recipientCounts,Ja.put(u,h),Qa.put(u,d)}d={},o||Object.assign(d,Qa.get(u));const p=new $s(t);for(let f=0;f<h.length;++f){const m=h[f],g=m.sourceLayer;if(g instanceof oa)continue;const x=[],b=[],w=[];let y=!1;for(const k of m.inputs){const S=p.getValue(k),C=p.getMask(k);x.push(S),b.push(C),C!=null&&(y=!0),o||(d[k.name]--,d[k.name]===0&&!t.hasKey(k)&&a.indexOf(k.name)===-1&&!S.isDisposed&&k.sourceLayer.stateful!==!0&&w.push(S))}y&&(e=e||{},e.mask=b[0]);const $=At(g.apply(x,e));let v=null;g.supportsMasking&&(v=g.computeMask(x,b));const N=HR(m),T=Array.isArray(N)?N:[N];for(let k=0;k<T.length;++k){p.hasKey(T[k])||p.add(T[k],$[k],Array.isArray(v)?v[0]:v);const S=a.indexOf(T[k].name);S!==-1&&(l[S]=$[k])}o||xt(w)}return p.disposeMasks(),r?l:l[0]}function UR(n,t){I(n!=null&&n.length>0,()=>"Expected at least one fetch, got none");let e=[],s={};if(n.length===1){const o=Xf(n[0],t);e=o.sorted,s=o.recipientMap}else{const o=new Set;for(const r of n){const{sorted:i,recipientMap:a}=Xf(r,t);for(const l of i)o.has(l.name)||(e.push(l),o.add(l.name));for(const l in a)s[l]==null&&(s[l]=new Set),a[l].forEach(c=>s[l].add(c))}}return{sorted:e,recipientCounts:GR(s)}}function GR(n){const t={};for(const e in n)t[e]=n[e].size;return t}function Xf(n,t){const e=new Set,s=[],o={};for(const a of t.names())e.add(a);const r=[],i=[];for(r.push(n);r.length>0;){const a=r[r.length-1];if(e.has(a.name)){r.pop();continue}const l=i[i.length-1]===r.length-1;if(a.inputs.length===0||l)r.pop(),s.push(a),e.add(a.name),l&&i.pop();else{i.push(r.length-1);for(const c of a.inputs)o[c.name]==null&&(o[c.name]=new Set),o[c.name].add(a.name),!e.has(c.name)&&r.push(c)}}return{sorted:s,recipientMap:o}}function HR(n){let t;if(n.sourceLayer.inboundNodes.length===1)t=n.sourceLayer.output;else{let e=null;for(let s=0;s<n.sourceLayer.inboundNodes.length;++s)for(const o of n.sourceLayer.inboundNodes[s].outputTensors)if(o.id===n.id){e=s;break}t=n.sourceLayer.getOutputAt(e)}return t}const qR=B();qR.registerFlag("TOPOLOGICAL_SORT_CACHE_MAX_ENTRIES",()=>100,WR);function Ip(n,t){return z(()=>Ne(it(A(n,n),t,!0)))}class ra extends No{getConfig(){return{}}}class Ib extends ra{constructor(t){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=t.maxValue!=null?t.maxValue:this.defaultMaxValue,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return z(()=>{const e=Ip(t,this.axis),s=on(e,0,this.maxValue);return A(t,ct(s,J(ie(),e)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}}Ib.className="MaxNorm";X(Ib);class vb extends ra{constructor(t){super(),this.defaultAxis=0,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return z(()=>ct(t,J(ie(),Ip(t,this.axis))))}getConfig(){return{axis:this.axis}}}vb.className="UnitNorm";X(vb);class kb extends ra{apply(t){return ds(t)}}kb.className="NonNeg";X(kb);class Sb extends ra{constructor(t){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=t.minValue!=null?t.minValue:this.defaultMinValue,this.maxValue=t.maxValue!=null?t.maxValue:this.defaultMaxValue,this.rate=t.rate!=null?t.rate:this.defaultRate,this.axis=t.axis!=null?t.axis:this.defaultAxis}apply(t){return z(()=>{const e=Ip(t,this.axis),s=J(A(this.rate,on(e,this.minValue,this.maxValue)),A(1-this.rate,e));return A(t,ct(s,J(ie(),e)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}}Sb.className="MinMaxNorm";X(Sb);const Yf={maxNorm:"MaxNorm",minMaxNorm:"MinMaxNorm",nonNeg:"NonNeg",unitNorm:"UnitNorm"};function le(n){return dp(n)}function Zf(n,t={}){return ta(n,tn.getMap().classNameMap,t,"constraint")}function ce(n){if(n==null)return null;if(typeof n=="string"){const e={className:n in Yf?Yf[n]:n,config:{}};return Zf(e)}else return n instanceof ra?n:Zf(n)}async function qs(n){if(n==null)return;const t=[],e=[],s=[];for(const o in n){const r=n[o];if(typeof r!="number"){const i=r;t.push(i.data()),e.push(o),s.push(i)}}if(t.length>0){const o=await Promise.all(t);for(let r=0;r<o.length;++r)n[e[r]]=o[r][0];xt(s)}}function Nb(n){if(n!=null)for(const t in n){const e=n[t];typeof e!="number"&&e.dispose()}}var Jf;(function(n){n[n.SILENT=0]="SILENT",n[n.VERBOSE=1]="VERBOSE"})(Jf||(Jf={}));const jR=125;class Pr{constructor(){this.validationData=null}setParams(t){this.params=t}async onEpochBegin(t,e){}async onEpochEnd(t,e){}async onBatchBegin(t,e){}async onBatchEnd(t,e){}async onTrainBegin(t){}async onTrainEnd(t){}setModel(t){}}class KR{constructor(t,e=10){t==null&&(t=[]),this.callbacks=t,this.queueLength=e}append(t){this.callbacks.push(t)}setParams(t){for(const e of this.callbacks)e.setParams(t)}setModel(t){for(const e of this.callbacks)e.setModel(t)}async onEpochBegin(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onEpochBegin(t,e)}async onEpochEnd(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onEpochEnd(t,e)}async onBatchBegin(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onBatchBegin(t,e)}async onBatchEnd(t,e){e==null&&(e={});for(const s of this.callbacks)await s.onBatchEnd(t,e)}async onTrainBegin(t){t==null&&(t={});for(const e of this.callbacks)await e.onTrainBegin(t)}async onTrainEnd(t){t==null&&(t={});for(const e of this.callbacks)await e.onTrainEnd(t)}}class XR extends Pr{constructor(){super()}async onEpochBegin(t){this.seen=0,this.totals={}}async onBatchEnd(t,e){e==null&&(e={});const s=e.size==null?0:e.size;this.seen+=s;for(const o in e){const r=e[o];if(typeof r=="number")this.totals.hasOwnProperty(o)||(this.totals[o]=0),this.totals[o]=this.totals[o]+r*s;else{let i;o in this.totals?i=this.totals[o]:this.totals[o]=0;const a=z(()=>J(this.totals[o],A(r,s)));this.totals[o]=a,i?.dispose()}}}async onEpochEnd(t,e){if(e!=null)for(const s of this.params.metrics)this.totals[s]!=null&&(typeof this.totals[s]=="number"?e[s]=this.totals[s]/this.seen:z(()=>{const o=A(ct(1,this.seen),this.totals[s]);e[s]=o,this.totals[s].dispose(),Bn(e[s])}))}}class YR extends Pr{async onTrainBegin(t){this.epoch=[],this.history={}}async onEpochEnd(t,e){e==null&&(e={}),this.epoch.push(t);for(const s in e)this.history[s]==null&&(this.history[s]=[]),this.history[s].push(e[s])}async syncData(){const t=[],e=[],s=[];for(const r in this.history){const i=this.history[r];for(let a=0;a<i.length;++a)if(typeof i[a]!="number"){const l=i[a];t.push(l.data()),e.push(r),s.push(a)}}const o=await Promise.all(t);for(let r=0;r<o.length;++r)this.history[e[r]][s[r]].dispose(),this.history[e[r]][s[r]]=o[r][0]}}class ZR extends Pr{constructor(t,e){if(super(),this.currentEpoch=0,this.nowFunc=t.nowFunc,this.nextFrameFunc=t.nextFrameFunc||Dx,this.yieldEvery=e||"auto",this.yieldEvery==="auto"&&(this.yieldEvery=jR),this.yieldEvery==="never"&&t.onYield!=null)throw new Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");Ba(this.yieldEvery)&&(this.maybeWait=uR(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=t.onTrainBegin,this.trainEnd=t.onTrainEnd,this.epochBegin=t.onEpochBegin,this.epochEnd=t.onEpochEnd,this.batchBegin=t.onBatchBegin,this.batchEnd=t.onBatchEnd,this.yield=t.onYield}async maybeWait(t,e,s){const o=[];this.yield!=null&&(await qs(s),o.push(this.yield(t,e,s))),o.push(this.nextFrameFunc()),await Promise.all(o)}async onEpochBegin(t,e){this.currentEpoch=t,this.epochBegin!=null&&(await qs(e),await this.epochBegin(t,e))}async onEpochEnd(t,e){const s=[];this.epochEnd!=null&&(await qs(e),s.push(this.epochEnd(t,e))),this.yieldEvery==="epoch"&&s.push(this.nextFrameFunc()),await Promise.all(s)}async onBatchBegin(t,e){this.batchBegin!=null&&(await qs(e),await this.batchBegin(t,e))}async onBatchEnd(t,e){const s=[];this.batchEnd!=null&&(await qs(e),s.push(this.batchEnd(t,e))),this.yieldEvery==="batch"?s.push(this.nextFrameFunc()):Ba(this.yieldEvery)&&s.push(this.maybeWait(this.currentEpoch,t,e)),await Promise.all(s)}async onTrainBegin(t){this.trainBegin!=null&&(await qs(t),await this.trainBegin(t))}async onTrainEnd(t){this.trainEnd!=null&&(await qs(t),await this.trainEnd(t))}}function Tb(n,t){return n==null&&(n={}),n instanceof Pr?[n]:Array.isArray(n)&&n[0]instanceof Pr?n:At(n).map(s=>new ZR(s,t))}class hn{constructor(){}static registerCallbackConstructor(t,e){I(t>=0&&Number.isInteger(t),()=>`Verbosity level is expected to be an integer >= 0, but got ${t}`),hn.checkForDuplicate(e),hn.constructors[t]==null&&(hn.constructors[t]=[]),hn.constructors[t].push(e)}static checkForDuplicate(t){for(const e in hn.constructors)hn.constructors[+e].forEach(o=>{if(o===t)throw new F("Duplicate callback constructor.")})}static clear(){hn.constructors={}}static createCallbacks(t){const e=[];for(const s in hn.constructors){const o=+s;t>=o&&e.push(...hn.constructors[o])}return e.map(s=>new s)}}hn.constructors={};function Eb(n,t,e,s,o,r,i,a,l){const c=new YR,u=[new XR,...hn.createCallbacks(t)];n!=null&&u.push(...n),u.push(c);const h=new KR(u);return h.setParams({epochs:e,initialEpoch:s,samples:o,steps:r,batchSize:i,verbose:t,doValidation:a,metrics:l}),{callbackList:h,history:c}}function Tn(n,t={},e=!1){return ta(n,tn.getMap().classNameMap,t,"layer",e)}function tl(n,t){return z(()=>{n.dtype!=="float32"&&(n=st(n,"float32"));const e=it(na(n),t,!0),s=qi(e.shape,ie()),o=Ne(zs(e,s));return ct(n,o)})}function _c(n,t){return z(()=>te(na(rt(t,n)),-1))}function vp(n,t){return z(()=>te(he(rt(t,n)),-1))}function kp(n,t){return z(()=>{const e=rt(n,t),s=on(he(n),ie(),Number.MAX_VALUE),o=he(ct(e,s));return A(100,te(o,-1))})}function JR(n,t){return z(()=>{const e=on(t,ie(),Number.MAX_VALUE),s=mn(J(1,e)),o=on(n,ie(),Number.MAX_VALUE),r=mn(J(1,o));return te(na(rt(s,r)),-1)})}function QR(n,t){return z(()=>{const e=zs(0,rt(1,A(n,t)));return te(na(e),-1)})}function tA(n,t){return z(()=>{const e=zs(0,rt(1,A(n,t)));return te(e,-1)})}function eA(n,t){return z(()=>{const e=it(A(n,t),-1),s=Nn(A(rt(1,n),t),-1);return zs(0,J(1,rt(s,e)))})}function nA(n,t){return z(()=>{const e=Math.log(2),s=rt(t,n),o=rt(J(s,ji(A(-2,s))),e);return te(o,-1)})}function zr(n,t,e=!1){return z(()=>{if(e)t=Sd(t);else{const s=it(t,t.shape.length-1,!0);t=ct(t,s)}return t=on(t,ie(),1-ie()),Ht(it(A(st(n,"float32"),mn(t)),t.shape.length-1))})}function el(n,t,e=!1){return z(()=>{const s=st(vc(CR(n)),"int32");t=on(t,ie(),1-ie());const o=t.shape,r=L(Xg(s,o[o.length-1]),o);return zr(r,t,e)})}function sA(n,t){if(!Tt(n.shape,t.shape))throw new F(`logits and labels must have the same shape, but got shapes ${JSON.stringify(n.shape)} and ${JSON.stringify(t.shape)}`);return z(()=>{const e=ds(t),s=Ht(he(t));return J(rt(e,A(t,n)),md(rn(s)))})}function Oc(n,t){return z(()=>{let e;return e=on(t,ie(),1-ie()),e=mn(ct(e,rt(1,e))),te(sA(n,e),-1)})}function oA(n,t){return z(()=>{const e=on(n,ie(),1),s=on(t,ie(),1);return it(A(n,mn(ct(e,s))),-1)})}function rA(n,t){return z(()=>{const e=mn(J(ie(),t));return te(rt(t,A(n,e)),-1)})}function Rb(n,t){return z(()=>{const e=tl(n,-1),s=tl(t,-1),o=A(e,s);return Ht(it(o,-1))})}const nl={meanSquaredError:_c,meanAbsoluteError:vp,meanAbsolutePercentageError:kp,meanSquaredLogarithmicError:JR,squaredHinge:QR,hinge:tA,categoricalHinge:eA,logcosh:nA,categoricalCrossentropy:zr,sparseCategoricalCrossentropy:el,binaryCrossentropy:Oc,kullbackLeiblerDivergence:oA,poisson:rA,cosineProximity:Rb};function au(n){if(typeof n=="string"){if(n in nl)return nl[n];let t=`Unknown loss ${n}`;throw n.toLowerCase().includes("softmaxcrossentropy")&&(t=`Unknown loss ${n}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new F(t)}else return n}function Ab(n,t){return z(()=>{const e=A(.5,gn(t)),s=Wn(Ze(t,e),n.dtype);return te(fn(n,s),-1)})}function Db(n,t){return z(()=>Wn(fn(Fr(n,-1),Fr(t,-1)),"float32"))}function Fb(n,t){return z(()=>st(it(Gn(fn(n,1),fn(t,1))),"float32"))}function iA(n,t){return z(()=>st(it(Gn(fn(n,1),fn(t,0))),"float32"))}function aA(n,t){return z(()=>st(it(Gn(fn(n,0),fn(t,1))),"float32"))}function lA(n,t){return z(()=>{const e=Fb(n,t),s=aA(n,t),o=J(e,s);return st(Se(Ze(o,0),ct(e,o),0),"float32")})}function a6(n,t){return z(()=>{const e=Fb(n,t),s=iA(n,t),o=J(e,s);return st(Se(Ze(o,0),ct(e,o),0),"float32")})}function cA(n,t){return Oc(n,t)}function uA(n,t){return n.rank===t.rank&&(n=Xi(n,[n.rank-1])),t=Fr(t,-1),t.dtype!==n.dtype&&(t=st(t,n.dtype)),st(fn(n,t),"float32")}function l6(n,t){return z(()=>{const e=n.sub(t).square().sum(),s=n.sub(n.mean()).square().sum();return mt(1).sub(e.div(s))})}const hA=_c,dA=_c,pA=vp,fA=vp,mA=kp,gA=kp,_b=zr,xA=Rb,Ob=el,sl={binaryAccuracy:Ab,categoricalAccuracy:Db,precision:lA,categoricalCrossentropy:_b,sparseCategoricalCrossentropy:Ob,mse:hA,MSE:dA,mae:pA,MAE:fA,mape:mA,MAPE:gA,cosine:xA};function bA(n){if(typeof n=="string"&&n in sl)return sl[n];if(typeof n!="string"&&n!=null)return n;throw new F(`Unknown metric ${n}`)}function ba(n){if(Mn(n!==null,`Unknown LossOrMetricFn ${n}`),typeof n=="string")return n;{let t;for(const e of Object.keys(nl))if(nl[e]===n){t=e;break}if(t!==void 0)return t;for(const e of Object.keys(sl))if(sl[e]===n){t=e;break}return t!==void 0?t:n.name}}function yA(n){const t={Adagrad:()=>_o.adagrad(.01),Adadelta:()=>_o.adadelta(1,.95,ie()),Adam:()=>_o.adam(.001,.9,.999,ie()),Adamax:()=>_o.adamax(.002,.9,.999,ie(),0),RMSProp:()=>_o.rmsprop(.001,.9,0,ie()),SGD:()=>_o.sgd(.01)};if(t.adagrad=t.Adagrad,t.adadelta=t.Adadelta,t.adam=t.Adam,t.adamax=t.Adamax,t.rmsprop=t.RMSProp,t.sgd=t.SGD,n in t)return t[n]();throw new F(`Unknown Optimizer ${n}`)}const Qf=1*1024*1024;function tm(n,t,e=!1){if(n==null||typeof n!="object"||Object.getPrototypeOf(n)!==Object.prototype||!Mu(n))throw new Error("User-defined metadata is expected to be a JSON object, but is not.");if(e){const s=JSON.stringify(n);s.length>Qf&&console.warn(`User-defined metadata of model "${t}" is too large in size (length=${s.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= ${Qf}.`)}}function Mu(n){if(n===null)return!0;if(typeof n=="object")if(Object.getPrototypeOf(n)===Object.prototype){const t=Object.keys(n);for(const e of t)if(typeof e!="string"||!Mu(n[e]))return!1;return!0}else if(Array.isArray(n)){for(const t of n)if(!Mu(t))return!1;return!0}else return!1;else{const t=typeof n;return t==="string"||t==="number"||t==="boolean"}}function wA(n,t,e,s=console.log){const o=CA(n),r=["Layer (type)","Input Shape","Output shape","Param #"];o?(t=t||90,e=e||[.32,.61,.89,1]):(t=t||115,e=e||[.24,.48,.7,.8,1]),e[e.length-1]<=1&&(e=e.map(u=>Math.floor(t*u)));let i;if(!o){r.push("Receives inputs"),i=[];for(const u in n.nodesByDepth)i.push(...n.nodesByDepth[u])}s("_".repeat(t)),ol(r,e,s),s("=".repeat(t));const a=n.layers;for(let u=0;u<a.length;++u)o?IA(a[u],e,s):vA(a[u],e,i,s),s((u===a.length-1?"=":"_").repeat(t));n.checkTrainableWeightsConsistency();const l=$A(n),c=Za(n.nonTrainableWeights);s(`Total params: ${l+c}`),s(`Trainable params: ${l}`),s(`Non-trainable params: ${c}`),s("_".repeat(t))}function $A(n){let t;return n.collectedTrainableWeights!=null?t=Za(n.collectedTrainableWeights):t=Za(n.trainableWeights),t}function CA(n){let t=!0;const e=[],s=[];for(const o in n.nodesByDepth)e.push(n.nodesByDepth[o]);for(const o of e){if(o.length>1||o.length===1&&o[0].inboundLayers.length>1){t=!1;break}s.push(...o)}if(t)for(const o of n.layers){let r=!1;for(const i of o.inboundNodes)if(s.indexOf(i)!==-1)if(r){t=!1;break}else r=!0;if(!t)break}return t}function ol(n,t,e=console.log){let s="";for(let o=0;o<n.length;++o)o>0&&(s=s.slice(0,s.length-1)+" "),s+=n[o],s=s.slice(0,t[o]),s+=" ".repeat(t[o]-s.length);e(s)}function IA(n,t,e){let s,o;try{o=n.inboundNodes.map(l=>JSON.stringify(l.inputShapes)).join(",")}catch{o="multiple"}try{s=JSON.stringify(n.outputShape)}catch{s="multiple"}const r=n.name,i=n.getClassName(),a=[`${r} (${i})`,o,s,n.countParams().toString()];ol(a,t,e)}function vA(n,t,e,s){let o,r;try{r=n.inboundNodes.map(h=>JSON.stringify(h.inputShapes)).join(",")}catch{r="multiple"}try{o=JSON.stringify(n.outputShape)}catch{o="multiple"}const i=[];for(const h of n.inboundNodes)if(!(e!=null&&e.length>0&&e.indexOf(h)===-1))for(let d=0;d<h.inboundLayers.length;++d){const p=h.inboundLayers[d].name,f=h.nodeIndices[d],m=h.tensorIndices[d];i.push(`${p}[${f}][${m}]`)}const a=n.name,l=n.getClassName(),c=i.length===0?"":i[0],u=[`${a} (${l})`,r,o,n.countParams().toString(),c];ol(u,t,s);for(let h=1;h<i.length;++h)ol(["","","","",i[h]],t,s)}function Lb(n,t,e){return(n==="inboundNodes"||n==="outputLayers"||n==="inputLayers")&&t===0&&typeof e=="string"}function Br(n,t){if(n===null)return null;if(typeof n=="string")return Ys(n);if(typeof n=="number"||typeof n=="boolean")return n;if(n instanceof Array){const e=[],s=n.length;for(let o=0;o<s;++o){const r=n[o];Lb(t,o,r)?e.push(r):e.push(Br(r,t))}return e}else{const e={};for(const s of Object.keys(n)){const o=n[s];if(s==="name"&&typeof o=="string")e[s]=o;else{const r=Ys(s);e[r]=Br(o,r)}}return e}}function Pu(n,t){if(n==null)return null;if(typeof n=="string")return Zn(n);if(typeof n=="number"||typeof n=="boolean")return n;if(n instanceof Array){const e=[],s=n.length;for(let o=0;o<s;++o){const r=n[o];Lb(t,o,r)?e.push(r):e.push(Pu(r,t))}return e}else{const e={};for(const s of Object.keys(n)){const o=n[s],r=Zn(s);(s==="name"||s==="className")&&typeof o=="string"?e[r]=o:e[r]=Pu(o,s)}return e}}const Mb="4.22.0";const kA=n=>{const t=Object.keys(n);if(t.length===0)return!1;const e=t[0].split("/");return!isNaN(parseInt(e[e.length-1],10))};class Cn extends It{constructor(t){if(super({}),this.containerNodes=new Set,this.name=t.name,this.name==null){const b=this.getClassName().toLowerCase();this.name=Ac(b)}if(this.supportsMasking=!1,this.trainable_=!0,Array.isArray(t.inputs)?this.inputs=t.inputs.slice():this.inputs=[t.inputs],Array.isArray(t.outputs)?this.outputs=t.outputs.slice():this.outputs=[t.outputs],ks(this.inputs).length!==this.inputs.length)throw new F(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(b=>b.name)}`);ks(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(b=>b.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[];for(const b of this.outputs){const w=b.sourceLayer,y=b.nodeIndex,$=b.tensorIndex;this.outputLayers.push(w),this.outputLayersNodeIndices.push(y),this.outputLayersTensorIndices.push($)}for(const b of this.inputs){const w=b.sourceLayer,y=b.nodeIndex,$=b.tensorIndex;Mn(y===0,"input layer has >1 nodes"),Mn($===0,"input layer has >1 tensors"),this.inputLayers.push(w),this.inputLayersNodeIndices.push(y),this.inputLayersTensorIndices.push($)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let b=0;b<this.inputLayers.length;b++){const w=this.inputLayers[b];if(!(w instanceof oa))throw new TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${t.inputs}. Input ${b} (0-based) originates from layer type ${w.getClassName()}.`);this.inputNames.push(w.name),this.feedInputShapes.push(w.batchInputShape),this.feedInputNames.push(w.name)}for(const b of this.outputLayers)this.outputNames.push(b.name);this.internalInputShapes=this.inputs.map(b=>b.shape),this.internalOutputShapes=this.outputs.map(b=>b.shape);const e={},s={},o={},r={},i={},a=[],l=(b,w,y,$,v,N)=>{($==null||v==null||N==null)&&($=b.sourceLayer,v=b.nodeIndex,N=b.tensorIndex);const T=$.inboundNodes[v];if(y.indexOf(T)!==-1)throw new dn(`The tensor ${b.name} at layer "${$.name}" is part of a cycle.`);if(w.indexOf(T)!==-1)return;this.containerNodes.add(Cn.nodeKey($,v)),$.id in i||(i[$.id]=Object.keys(i).length),y.indexOf(T)===-1&&y.push(T);const k=T.inboundLayers.length;for(let S=0;S<k;S++){const C=T.inputTensors[S],D=T.inboundLayers[S],O=T.nodeIndices[S],P=T.tensorIndices[S];l(C,w,y,D,O,P)}for(w.push(T);y.indexOf(T)>=0;)y.splice(y.indexOf(T),1);a.push(T)},c=[],u=[];for(const b of this.outputs)l(b,c,u);const h=a.slice().reverse();for(const b of h){s[b.id]=b,b.id in e||(e[b.id]=0);let w=e[b.id];const y=o[b.outboundLayer.id]==null?0:o[b.outboundLayer.id];w=Math.max(w,y),o[b.outboundLayer.id]=w,r[b.outboundLayer.id]=b.outboundLayer,e[b.id]=w;for(let $=0;$<b.inboundLayers.length;$++){const v=b.inboundLayers[$],N=b.nodeIndices[$],T=v.inboundNodes[N],k=e[T.id]==null?0:e[T.id];e[T.id]=Math.max(w+1,k),s[T.id]=T}}const d={};for(const b in e){const w=e[b];w in d||(d[w]=[]),d[w].push(s[b])}const p={};for(const b in o){const w=o[b];w in p||(p[w]=[]),p[w].push(r[b])}let f=Object.keys(p).map(b=>parseInt(b,10)).sort(ma);this.layers=[];for(const b of f){const w=p[b];w.sort((y,$)=>{const v=i[y.id],N=i[$.id];return v<N?-1:v>N?1:0});for(const y of w)y instanceof Cn&&this.internalContainerRefs.push(y),this.layers.push(y)}this.layersByDepth=p,f=Object.keys(d).map(b=>parseInt(b,10)).sort(ma);const m=this.inputs.slice(),g=[];for(const b of f)for(const w of d[b]){const y=w.outboundLayer;if(y!=null){for(const $ of w.inputTensors)if(m.indexOf($)===-1)throw new dn(`Graph disconnected: cannot obtain value for tensor ${$} at layer "${y.name}". The following previous layers were accessed without issue: ${g}`);for(const $ of w.outputTensors)m.push($);g.push(y.name)}}this.nodesByDepth=d;const x=this.layers.map(b=>b.name);for(const b of x){const w=x.filter(y=>y===b).length;if(w!==1)throw new dn(`The name "${b}" is used ${w} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(x))}this.outboundNodes=[],this.inboundNodes=[],new Fc({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(b=>null),outputMasks:this.outputs.map(b=>null),inputShapes:this.inputs.map(b=>b.shape),outputShapes:this.outputs.map(b=>b.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(this._refCount===0)throw new Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();const t={refCountAfterDispose:null,numDisposedVariables:0};if(--this._refCount===0){for(const e of this.layers)t.numDisposedVariables+=e.dispose().numDisposedVariables;for(const e of this.internalContainerRefs)t.numDisposedVariables+=e.dispose().numDisposedVariables}return t.refCountAfterDispose=this._refCount,t}get trainable(){return this.trainable_}set trainable(t){this.layers.forEach(e=>{e._trainableWeights.forEach(s=>s.trainable=t)}),this.trainable_=t}get trainableWeights(){if(this._trainableWeights.length>0)throw new F("Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.");if(!this.trainable)return[];let t=[];for(const e of this.layers)t=t.concat(e.trainableWeights);return t}get nonTrainableWeights(){const t=[];for(const e of this.layers)t.push(...e.nonTrainableWeights);if(!this.trainable){const e=[];for(const s of this.layers)e.push(...s.trainableWeights);return e.concat(t)}return t}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(t,e=!0){const s={};let o=0;const r=kA(t);r&&this.parseWeights(t);for(const a of this.layers)for(const[l,c]of a.weights.entries()){const u=r?`${c.name.split("/").slice(0,-1).join("/")+"/"}${l}`:c.originalName;if(s[u]!=null)throw new F(`Duplicate weight name: ${u}`);s[u]=c,o++}const i=[];for(const a in t){let l=a;if(s[a]==null){const c=a.split("/");l=c.slice(0,-2).concat([c[c.length-1]]).join("/")}if(s[l]!=null)i.push([s[l],t[a]]);else if(e)throw new F(`Provided weight data has no target variable: ${a}`);delete s[l]}if(e){const a=[];for(const l in s)a.push(l);if(a.length>0)throw new F(`${a.length} of ${o} weights are not set: ${a}`)}Cp(i)}parseWeights(t){for(const e in Object.keys(t)){const s=e.split("/"),o=["vars","layer_checkpoint_dependencies"],r=s.map(i=>i.startsWith("_")?i.slice(1):i).filter(i=>!o.includes(i)).join("/");r!==e&&(t[r]=t[e],delete t[e])}}updatedConfig(){const t=this.getConfig(),e={};return e.className=this.getClassName(),e.config=t,e.kerasVersion=`tfjs-layers ${Mb}`,e.backend="TensorFlow.js",e}toJSON(t,e=!0){const s=Pu(this.updatedConfig());return e?JSON.stringify(s):s}call(t,e){return z(()=>{t=At(t);const s=new $s;for(let o=0;o<this.inputs.length;++o)s.add(this.inputs[o],t[o]);return Cr(this.outputs,s,e)})}computeMask(t,e){return z(()=>{t=At(t);let s;return e==null?s=bo(null,t.length):s=At(e),this.runInternalGraph(t,s)[1]})}computeOutputShape(t){const e=Ya(t);if(e.length!==this.inputLayers.length)throw new F(`Invalid inputShape argument ${t}: model has ${this.inputLayers.length} tensor inputs.`);const s={};for(let a=0;a<e.length;a++){const l=this.inputLayers[a],c=e[a],u=l.name+"_0_0";s[u]=c}const o=Object.keys(this.nodesByDepth).map(a=>parseInt(a,10)).sort(ma);if(o.length>1)for(const a of o){const l=this.nodesByDepth[a];for(const c of l){const u=c.outboundLayer;if(this.inputLayers.map(m=>m.id).indexOf(u.id)!==-1)continue;const h=[];for(let m=0;m<c.inboundLayers.length;m++){const g=c.inboundLayers[m],x=c.nodeIndices[m],b=c.tensorIndices[m],w=`${g.name}_${x}_${b}`,y=s[w];h.push(y)}const d=u.computeOutputShape(Pe(h)),p=Ya(d),f=u.inboundNodes.indexOf(c);for(let m=0;m<p.length;m++){const g=`${u.name}_${f}_${m}`;s[g]=p[m]}}}const r=[],i=[];for(let a=0;a<this.outputLayers.length;a++){const l=this.outputLayers[a],c=this.outputLayersNodeIndices[a],u=this.outputLayersTensorIndices[a],h=`${l.name}_${c}_${u}`;i.push(h)}for(let a=0;a<i.length;a++){const l=i[a];Mn(l in s),r.push(s[l])}return Pe(r)}runInternalGraph(t,e){e==null&&(e=bo(null,t.length));const s={};for(let l=0;l<this.inputs.length;++l){const c=this.inputs[l],u=t[l],h=e[l];s[c.id]=[u,h]}const o=Object.keys(this.nodesByDepth).map(l=>parseInt(l,10)).sort(ma);for(const l of o){const c=this.nodesByDepth[l];for(const u of c){const h=u.outboundLayer,d=u.inputTensors,p=u.outputTensors,f=new Array;for(const m of d)m.id in s&&f.push(s[m.id]);if(f.length===d.length){let m={},g,x,b,w;if(u.callArgs!=null&&(m=u.callArgs),f.length===1){const[y,$]=f[0];m.mask==null&&(m.mask=$),b=At(h.call(y,m)),w=At(h.computeMask(y,$)),g=[y],x=[$]}else g=f.map(y=>y[0]),x=f.map(y=>y[1]),m.mask==null&&(m.mask=x),b=At(h.call(g,m)),w=At(h.computeMask(g,x));if(h.activityRegularizer)throw new yt("LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.");for(let y=0;y<p.length;++y){const $=p[y],v=b[y],N=w[y];s[$.id]=[v,N]}}}}const r=[],i=[],a=[];for(const l of this.outputs){Mn(l.id in s,`Could not compute output ${l.name} : ${l.id}`);const[c,u]=s[l.id];a.push(c.shape),r.push(c),i.push(u)}return[r,i,a]}buildNodeConversionMap(t){const e={};let s;for(const o of this.layers){s=o instanceof Cn?1:0;for(let r=0;r<o.inboundNodes.length;r++){const i=Cn.nodeKey(o,r);this.containerNodes.has(i)&&(e[i]=s,s+=1)}}return e}getLayer(t,e){if(e!=null)return this.findLayer(e);if(t==null)throw new F("Provide either a layer name or layer index");if(typeof t=="number")return this.findLayer(t);for(const s of this.layers)if(s.name===t)return s;throw new F(`No such layer: ${t}`)}findLayer(t){if(this.layers.length<=t)throw new F(`Was asked to retrieve layer at index ${t}, but model only has ${this.layers.length} layer(s).`);return this.layers[t]}calculateLosses(){return z(()=>{const t=[];for(const e of this.layers)for(let s=0;s<e.inboundNodes.length;++s){const o=Cn.nodeKey(e,s);this.containerNodes.has(o)&&t.push(...e.calculateLosses())}return t})}getConfig(){const t={name:this.name},e=this.buildNodeConversionMap(this.layers),s=[];for(const i of this.layers){const a=i.getClassName(),l=i.getConfig(),c=[];for(let h=0;h<i.inboundNodes.length;h++){const d=i.inboundNodes[h],p=Cn.nodeKey(i,h);let f={};if(this.containerNodes.has(p)){if(d.callArgs)try{JSON.stringify(d.callArgs),f=d.callArgs}catch{console.warn(`Layer ${i.name} was passed non-serializable keyword arguments: ${d.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),f={}}if(d.inboundLayers.length>0){const m=[];for(let g=0;g<d.inboundLayers.length;g++){const x=d.inboundLayers[g],b=d.nodeIndices[g],w=d.tensorIndices[g],y=Cn.nodeKey(x,b);let $=e[y];$==null&&($=0),m.push([x.name,$,w,f])}c.push(m)}}}const u={};u.name=i.name,u.className=a,u.config=l,u.inboundNodes=c,s.push(u)}t.layers=s;const o=[];for(let i=0;i<this.inputLayers.length;i++){const a=this.inputLayers[i],l=this.inputLayersNodeIndices[i],c=Cn.nodeKey(a,l);if(!this.containerNodes.has(c))continue;let u=e[c];u==null&&(u=0);const h=this.inputLayersTensorIndices[i];o.push([a.name,u,h])}t.inputLayers=o;const r=[];for(let i=0;i<this.outputLayers.length;i++){const a=this.outputLayers[i],l=this.outputLayersNodeIndices[i],c=Cn.nodeKey(a,l);if(!this.containerNodes.has(c))continue;let u=e[c];u==null&&(u=0);const h=this.outputLayersTensorIndices[i];r.push([a.name,u,h])}return t.outputLayers=r,t}static fromConfig(t,e,s={},o=!1){const r={},i={};function a(g,x){g.name in i?i[g.name].push(x):i[g.name]=[x]}function l(g,x){const b=[];let w;for(const y of x){const $=y[0],v=y[1],N=y[2];if(w=y[3]==null?{}:y[3],!($ in r)){a(g,x);return}const T=r[$];if(T.inboundNodes.length<=v){a(g,x);return}const k=T.inboundNodes[v];b.push(k.outputTensors[N])}b.length>0&&g.apply(Pe(b),w)}function c(g){const x=g.name,b=Tn(g,e.customObjects!=null?e.customObjects:{});b.setFastWeightInitDuringBuild(o),r[x]=b,g.inboundNodes.forEach(y=>{if(!(y instanceof Array))throw new F(`Corrupted configuration, expected array for nodeData: ${y}`);a(b,y)})}const u=e.name,h=e.layers;for(const g of h)c(g);for(;!cR(i);)for(const g of h){const x=r[g.name];if(x.name in i){const b=i[x.name];delete i[x.name];for(const w of b)l(x,w)}}const d=[],p=[],f=e.inputLayers;for(const g of f){const x=g[0],b=g[1],w=g[2];Mn(x in r);const $=r[x].inboundNodes[b].outputTensors;d.push($[w])}const m=e.outputLayers;for(const g of m){const x=g[0],b=g[1],w=g[2];Mn(x in r);const $=r[x].inboundNodes[b].outputTensors;p.push($[w])}return new t({inputs:d,outputs:p,name:u})}get stateful(){if(this._stateful)throw new F("Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.");for(const t of this.layers)if(t.stateful)return!0;return!1}resetStates(){z(()=>{this.layers.forEach(t=>{t.stateful&&t.resetStates()})})}}function SA(n,t,e){const s=t.length;if(n==null||Array.isArray(n)&&n.length===0)return t.map(o=>null);if(s===1)return Array.isArray(n)&&n.length===1?n:typeof n=="object"&&t[0]in n?[n[t[0]]]:[n];if(Array.isArray(n)){if(n.length!==s)throw new Error(`Provided ${e} is an array of ${n.length} element(s), but the model has ${s} outputs. Make sure a set of weights is provided for each model output.`);return n}else if(typeof n=="object"&&Object.keys(n).length>0&&typeof n[Object.keys(n)[0]]=="object"){const o=[];return t.forEach(r=>{r in n?o.push(n[r]):o.push(null)}),o}else throw new Error(`The model has multiple (${s}) outputs, so ${e} must be either an array with ${s} elements or an object with ${t} keys. Provided ${e} not understood: ${JSON.stringify(n)}`)}function Pb(n,t){return SA(n,t,"classWeight")}async function zb(n,t,e,s){if(e!=null){const o=z(()=>{if(n.shape.length===1)return eo(n);if(n.shape.length===2){if(n.shape[1]>1)return Fr(n,1);if(n.shape[1]===1)return L(n,[n.shape[0]]);throw new Error(`Encountered unexpected last-dimension size (${n.shape[1]}) during handling of class weights. The size is expected to be >= 1.`)}else throw new Error(`Unexpected rank of target (y) tensor (${n.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),r=Array.from(await o.data());xt(o);const i=[];return r.forEach(a=>{if(e[a]==null)throw new Error(`classWeight must contain all classes in the training data. The class ${a} exists in the data but not in classWeight`);i.push(e[a])}),Be(i,"float32")}else return null}function NA(n,t){return A(n,t)}const TA=32;function Bb(n,t){let e,s;const o=t;e=o.xs,s=o.ys,I(e!=null&&s!=null,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${t}`);const r=em("input",n.inputNames,e),i=em("output",n.outputNames,s),a=r[0].shape[0];I(r.length===n.inputs.length,()=>`LayersModel has ${n.inputs.length} inputs, but the dataset provides ${r.length} inputs.  (Expected input keys: ${JSON.stringify(n.inputNames)})`),I(i.length===n.outputs.length,()=>`LayersModel has ${n.outputs.length} outputs, but the dataset provides ${i.length} outputs.  (Expected output keys: ${JSON.stringify(n.outputNames)})`);for(let l=0;l<r.length;l++)I(r[l].shape[0]===a,()=>`Batch size mismatch: input ${n.inputNames[l]} has ${r[l].shape[0]}; expected  ${a} based on input ${n.inputNames[0]}.`);for(let l=0;l<i.length;l++)I(i[l].shape[0]===a,()=>`Batch size mismatch: output ${n.outputNames[l]} has ${i[l].shape[0]}; expected  ${a} based on input ${n.inputNames[0]}.`);return{xs:r,ys:i}}function em(n,t,e){if(e instanceof Lt)return[e];if(Array.isArray(e))return I(e.length===t.length,()=>`Received an array of ${e.length} Tensors, but expected ${t.length} to match the ${n} keys ${t}.`),e;{const s=[];for(const o of t){if(e[o]==null)throw new F(`The feature data generated by the dataset lacks the required ${n} key '${o}'.`);s.push(e[o])}return s}}function EA(n){if(n.length===3)throw new yt("Validation with sample weights is not implemented yet.");return{xs:n[0],ys:n[1]}}async function RA(n,t,e){const s=e.batchesPerEpoch!=null;if(I(n.optimizer!=null,()=>"You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig)."),I(e!=null,()=>"For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call."),I(e.epochs!=null&&e.epochs>0&&Number.isInteger(e.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${e.epochs}`),I(!s||e.batchesPerEpoch>0&&Number.isInteger(e.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${e.batchesPerEpoch}`),I(e.validationSplit==null,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),n.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");n.isTraining=!0;try{const o=e.validationData!=null;let r,i;if(o)if(nm(e.validationData))I(e.validationBatches==null||e.validationBatches>0&&Number.isInteger(e.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${e.validationBatches}`);else{const g=EA(e.validationData);r=g.xs,i=g.ys}const a=n.makeTrainFunction(),l=n.getDedupedMetricsNames();let c;o?c=l.slice().concat(l.map(g=>"val_"+g)):c=l.slice();const u=Tb(e.callbacks,e.yieldEvery),h=e.verbose==null?1:e.verbose,{callbackList:d,history:p}=Eb(u,h,e.epochs,null,null,AA(t,e),null,o,c);d.setModel(n),n.history=p,await d.onTrainBegin(),n.stopTraining_=!1;let f=e.initialEpoch==null?0:e.initialEpoch,m=await t.iterator();for(;f<e.epochs;){const g={};await d.onEpochBegin(f);let x=0,b=0;for(s||(m=await t.iterator());!s||x<e.batchesPerEpoch;){const w=await m.next();if(s&&w.done){console.warn(`You provided \`batchesPerEpoch\` as ${e.batchesPerEpoch}, but your dataset iterator ran out of data after ${x} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${e.batchesPerEpoch*e.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(w.value!=null){const{xs:y,ys:$}=Bb(n,w.value),v={};v.batch=b,v.size=y[0].shape[0],await d.onBatchBegin(b,v);const N=[];if(e.classWeight!=null){const S=Pb(e.classWeight,n.outputNames);for(let C=0;C<S.length;++C)N.push(await zb($[C],null,S[C]))}const T=y.concat($).concat(N),k=a(T);xt(T);for(let S=0;S<l.length;++S){const C=l[S],D=k[S];v[C]=D,Bn(D)}await d.onBatchEnd(b,v),Nb(v),b++,x++}if(s?x>=e.batchesPerEpoch:w.done){if(o){let y;nm(e.validationData)?y=At(await n.evaluateDataset(e.validationData,{batches:e.validationBatches})):y=At(n.evaluate(r,i,{batchSize:e.validationBatchSize==null?TA:e.validationBatchSize,verbose:0}));for(let $=0;$<n.metricsNames.length;++$)g[`val_${n.metricsNames[$]}`]=y[$]}break}if(n.stopTraining_)break}if(await d.onEpochEnd(f,g),f++,n.stopTraining_)break}return await d.onTrainEnd(),await n.history.syncData(),n.history}finally{n.isTraining=!1}}function AA(n,t){let e=null;return t.batchesPerEpoch!=null?e=t.batchesPerEpoch:Number.isFinite(n.size)&&(e=n.size),e}function nm(n){return typeof n.iterator=="function"}function DA(n){return typeof n.next=="function"}async function FA(n,t,e){e=e||{};const s=e.batches!=null,o=n.testFunction;let r=[];if(e.verbose>0)throw new yt("Verbose mode is not implemented yet.");I(!s||e.batches>0&&Number.isInteger(e.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(e.batches)}`);const i=DA(t)?t:await t.iterator();let a=0,l=0;for(;!s||l<e.batches;){const c=await i.next();if(r=z(()=>{if(c.value){const{xs:u,ys:h}=Bb(n,c.value),d=u.concat(h),p=z(()=>o(d));if(xt(d),l===0)for(let m=0;m<p.length;++m)r.push(mt(0));const f=d[0].shape[0];for(let m=0;m<p.length;++m){const g=p[m],x=r[m];r[m]=z(()=>J(r[m],A(f,g))),l>0&&xt(x)}xt(p),a+=f,++l}return r}),c.done){s&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${e.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let c=0;c<r.length;++c){const u=r[c];r[c]=ct(r[c],a),xt(u)}return Pe(r)}function lu(n){I(n>0&&Number.isInteger(n),()=>`batchSize is required to be a positive integer, but got ${n}`)}function br(n,t,e){return n==null?[null]:Array.isArray(n)?n.map(s=>so(s,t,e-t)):so(n,t,e-t)}function zu(n,t){return z(()=>n==null?null:Array.isArray(n)?n.map(e=>zu(e,t)):pb(n,t.dtype==="int32"?t:st(t,"int32")))}function cu(n,t){const e=[];let s=0,o=null;for(;s<n;)o=s+t,o>=n&&(o=n),e.push([s,o]),s=o;return e}function Vb(n){const t=[];n instanceof Lt&&(n=[n]);for(let e=0;e<n.length;++e){const s=n[e];if(s.rank===1)t.push(ea(s,1));else{if(s.rank===0)throw new Error("Expected tensor to be at least 1D, but received a 0D tensor (scalar).");t.push(s)}}return t}function $n(n,t){if(n==null)return;const e=[];if(t instanceof Lt)e.push(t.id);else if(Array.isArray(t))t.forEach(o=>e.push(o.id));else if(t!=null)for(const o in t){const r=t[o];e.push(r.id)}const s=[];if(n instanceof Lt)e.indexOf(n.id)===-1&&s.push(n);else if(Array.isArray(n))n.forEach(o=>{e.indexOf(o.id)===-1&&s.push(o)});else if(n!=null)for(const o in n){const r=n[o];e.indexOf(r.id)===-1&&s.push(r)}s.forEach(o=>{o.isDisposed||o.dispose()})}function _A(n){return n instanceof Lt}function Bu(n){return Array.isArray(n)}function sm(n){return!_A(n)&&!Bu(n)}function om(n,t,e,s=!0,o=""){if(t==null||t.length===0){if(n!=null){let i=!1;if(Bu(n)&&n.length>0)i=!0;else if(sm(n)){for(const a in n)if(n.hasOwnProperty(a)){i=!0;break}}else i=!0;if(i)throw new F(`Error when checking model ${o} expected no data, but got ${n}`)}return[]}if(n==null)return t.map(i=>null);let r;if(sm(n)){n=n,r=[];for(const i of t){if(n[i]==null)throw new F(`No data provided for "${i}". Need data for each key in: ${t}`);r.push(n[i])}}else if(Bu(n)){if(n=n,n.length!==t.length)throw new F(`Error when checking model ${o}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${t.length} Tensor(s), but instead got the following list of Tensor(s): ${n}`);r=n}else{if(n=n,t.length>1)throw new F(`The model ${o} expects ${t.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${n.shape}`);r=[n]}if(r=Vb(r),e!=null)for(let i=0;i<t.length;++i){if(e[i]==null)continue;const a=r[i];if(a.shape.length!==e[i].length)throw new F(`Error when checking ${o}: expected ${t[i]} to have ${e[i].length} dimension(s). but got array with shape ${a.shape}`);for(let l=0;l<e[i].length;++l){if(l===0&&!s)continue;const c=a.shape[l],u=e[i][l];if(u!=null&&u>=0&&c!==u)throw new F(`${o} expected a batch of elements where each example has shape [${e[i].slice(1,e[i].length)}] (i.e.,tensor shape [*,${e[i].slice(1,e[i].length)}]) but the ${o} received an input with ${a.shape[0]} examples, each with shape [${a.shape.slice(1,a.shape.length)}] (tensor shape [${a.shape}])`)}}return r}function OA(n,t,e){const s=ks(n.map(r=>r.shape[0]));s.sort();const o=ks(t.map(r=>r.shape[0]));if(o.sort(),s.length>1)throw new F(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(n.map(r=>r.shape))}`);if(o.length>1)throw new F(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(t.map(r=>r.shape))}`);if(s.length>0&&o.length>0&&!Tt(s,o))throw new F(`Input Tensors should have the same number of samples as target Tensors. Found ${s[0]} input sample(s) and ${o[0]} target sample(s).`)}function LA(n,t,e){const s=[_c,Oc,zr];for(let o=0;o<n.length;++o){const r=n[o],i=t[o],a=e[o];if(i!=null){if(i===zr&&r.shape[r.shape.length-1]===1)throw new F(`You are passing a target array of shape ${r.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(s.indexOf(i)!==-1){const l=r.shape.slice(1),c=a.slice(1);for(let u=0;u<l.length;++u){const h=l[u],d=c[u];if(d!=null&&h!==d)throw new F(`A target Tensor with shape ${r.shape} was passed for an output of shape ${a}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}function rm(n,t,e,s=!0,o=""){let r;if(Array.isArray(n)){if(n.length!==t.length)throw new F(`Error when checking model ${o}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${t.length} Tensor(s), but instead got ${n.length} Tensors(s).`);r=n}else{if(t.length>1)throw new F(`The model expects ${t.length} ${o} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(n.shape)}.`);r=[n]}if(e!=null)for(let i=0;i<t.length;++i){if(e[i]==null)continue;const a=r[i];if(a.shape.length!==e[i].length)throw new F(`Error when checking ${o}: expected ${t[i]} to have ${e[i].length} dimension(s), but got array with shape ${JSON.stringify(a.shape)}`);for(let l=0;l<e[i].length;++l){if(l===0&&!s)continue;const c=a.shape[l],u=e[i][l];if(u!=null&&u!==c)throw new F(`Error when checking ${o}: expected ${t[i]} to have shape ${JSON.stringify(e[i])} but got array with shape ${JSON.stringify(a.shape)}.`)}}}function MA(n,t){if(n==null||Array.isArray(n)&&n.length===0)return t.map(s=>[]);let e;if(typeof n=="string"||typeof n=="function")e=[n];else if(Array.isArray(n)||typeof n=="object")e=n;else throw new TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${n}`);if(Array.isArray(e))return t.map(s=>e);{const s=[];for(const o of t){let r=e.hasOwnProperty(o)?e[o]:[];Array.isArray(r)||(r=[r]),s.push(r)}return s}}const PA="layers-model";class Uo extends Cn{constructor(t){super(t),this.isTraining=!1}summary(t,e,s=console.log){if(!this.built)throw new F("This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).");wA(this,t,e,s)}compile(t){if(t.loss==null&&(t.loss=[]),this.loss=t.loss,typeof t.optimizer=="string")this.optimizer_=yA(t.optimizer),this.isOptimizerOwned=!0;else{if(!(t.optimizer instanceof Vs))throw new F("User-defined optimizer must be an instance of tf.Optimizer.");this.optimizer_=t.optimizer,this.isOptimizerOwned=!1}let e=[];if(!Array.isArray(t.loss)&&typeof t.loss!="string"&&typeof t.loss!="function"){t.loss=t.loss;for(const i in t.loss)if(this.outputNames.indexOf(i)===-1)throw new F(`Unknown entry in loss dictionary: "${i}". Only expected the following keys: ${this.outputNames}`);for(const i of this.outputNames)t.loss[i]==null&&console.warn(`Output "${i}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${i} during training`),e.push(au(t.loss[i]))}else if(Array.isArray(t.loss)){if(t.loss.length!==this.outputs.length)throw new F(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${t.loss}.`);e=t.loss.map(a=>au(a))}else{const i=au(t.loss);this.outputs.forEach(a=>{e.push(i)})}this.lossFunctions=e,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let i=0;i<this.outputs.length;++i){const a=this.internalOutputShapes[i],l=this.outputNames[i];this.feedOutputNames.push(l),this.feedOutputShapes.push(a),this.feedLossFns.push(this.lossFunctions[i])}const s=[];this.metrics=t.metrics,this.metricsNames=["loss"],this.metricsTensors=[],no("loss",()=>{for(let i=0;i<this.outputs.length;++i){if(s.indexOf(i)!==-1)continue;const a=this.lossFunctions[i];this.outputs.length>1&&(this.metricsTensors.push([a,i]),this.metricsNames.push(this.outputNames[i]+"_loss"))}});const o=MA(t.metrics,this.outputNames),r=(i,a,l)=>{this.outputNames.length>1&&(a=this.outputNames[i]+"_"+a),this.metricsNames.push(a),this.metricsTensors.push([l,i])};no("metric",()=>{for(let i=0;i<this.outputs.length;++i){if(s.indexOf(i)!==-1)continue;const a=o[i];(c=>{let h,d,p;for(const f of c){if(typeof f=="string"&&["accuracy","acc","crossentropy","ce"].indexOf(f)!==-1){const g=this.internalOutputShapes[i];g[g.length-1]===1||this.lossFunctions[i]===Oc?["accuracy","acc"].indexOf(f)!==-1?d=Ab:["crossentropy","ce"].indexOf(f)!==-1&&(d=cA):this.lossFunctions[i]===el?["accuracy","acc"].indexOf(f)!==-1?d=uA:["crossentropy","ce"].indexOf(f)!==-1&&(d=Ob):["accuracy","acc"].indexOf(f)!==-1?d=Db:["crossentropy","ce"].indexOf(f)!==-1&&(d=_b);let x;["accuracy","acc"].indexOf(f)!==-1?x="acc":["crossentropy","ce"].indexOf(f)!==-1&&(x="ce"),p=d,h=""+x}else p=bA(f),h=""+ba(f);let m;no(h,()=>{m=p}),r(i,h,m)}})(a)}}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){this.collectedTrainableWeights!=null&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(t,e,s={}){const o=s.batchSize==null?32:s.batchSize;lu(o);const i=this.standardizeUserDataXY(t,e,!0,o);try{const a=i[0].concat(i[1]);this.makeTestFunction();const l=this.testFunction,c=this.testLoop(l,a,o,s.verbose,s.steps);return Pe(c)}finally{$n(i[0],t),$n(i[1],e)}}async evaluateDataset(t,e){return this.makeTestFunction(),FA(this,t,e)}checkNumSamples(t,e,s,o="steps"){let r;if(s!=null){if(r=null,e!=null)throw new F(`If ${o} is set, batchSize must be null or undefined.Got batchSize = ${e}`)}else if(t!=null)Array.isArray(t)?r=t[0].shape[0]:r=t.shape[0];else throw new F(`Either the input data should have a defined shape, or ${o} shoud be specified.`);return r}execute(t,e){if(Array.isArray(e)&&e.length===0)throw new F("`outputs` is an empty Array, which is not allowed.");const s=Array.isArray(e),o=s?e:[e],r=this.retrieveSymbolicTensors(o),i=new $s;if(t instanceof Lt&&(t=[t]),Array.isArray(t)){if(t.length!==this.inputs.length)throw new F(`The number of inputs provided (${t.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let l=0;l<this.inputs.length;++l)i.add(this.inputs[l],t[l])}else for(const l of this.inputs){const c=t[l.name];if(c==null)throw new F(`No value is provided for the model's input ${l.name}`);i.add(l,c)}const a=Cr(r,i);return s?a:a[0]}retrieveSymbolicTensors(t){const e=bo(null,t.length);let s=t.length;for(const o of this.layers){const r=Array.isArray(o.output)?o.output:[o.output],i=r.map(a=>a.name);for(let a=0;a<t.length;++a){const l=i.indexOf(t[a]);if(l!==-1&&(e[a]=r[l],s--),s===0)break}if(s===0)break}if(s>0){const o=[];throw e.forEach((r,i)=>{r==null&&o.push(t[i])}),new F(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(o)}`)}return e}predictLoop(t,e=32,s=!1){return z(()=>{const o=this.checkNumSamples(t);if(s)throw new yt("Verbose predictLoop() is not implemented yet.");const r=cu(o,e),i=this.outputs.map(a=>[]);for(let a=0;a<r.length;++a)z(()=>{const c=r[a][0],u=r[a][1],h=br(t,c,u),d=[];if(Array.isArray(h))for(let f=0;f<h.length;++f)d.push({key:this.inputs[f],value:h[f]});else d.push({key:this.inputs[0],value:h});const p=new $s(d);return Cr(this.outputs,p)}).forEach((c,u)=>i[u].push(c));return Pe(i.map(a=>ke(a,0)))})}predict(t,e={}){const s=Vb(t);rm(s,this.inputNames,this.feedInputShapes,!1);try{const o=e.batchSize==null?32:e.batchSize;return lu(o),this.predictLoop(s,o)}finally{$n(s,t)}}predictOnBatch(t){rm(t,this.inputNames,this.feedInputShapes,!0);const e=(Array.isArray(t)?t[0]:t).shape[0];return this.predictLoop(t,e)}standardizeUserDataXY(t,e,s=!0,o){if(this.optimizer_==null)throw new dn("You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).");const r=[];for(let i=0;i<this.feedOutputShapes.length;++i){const a=this.feedOutputShapes[i];this.feedLossFns[i]===el?r.push(a.slice(0,a.length-1).concat([1])):r.push(a)}if(t=om(t,this.feedInputNames,this.feedInputShapes,!1,"input"),e=om(e,this.feedOutputNames,r,!1,"target"),OA(t,e),LA(e,this.feedLossFns,this.feedOutputShapes),this.stateful&&o!=null&&o>0&&t[0].shape[0]%o!==0)throw new F(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${o}. Found: ${t[0].shape[0]} sample(s).`);return[t,e]}async standardizeUserData(t,e,s,o,r=!0,i){const[a,l]=this.standardizeUserDataXY(t,e,r,i);if(s!=null)throw new Error("sample weight is not supported yet.");let c=null;if(o!=null){const u=Pb(o,this.outputNames);c=[];for(let h=0;h<u.length;++h)c.push(await zb(l[h],null,u[h]))}return[a,l,c]}testLoop(t,e,s,o=0,r){return z(()=>{const i=this.checkNumSamples(e,s,r,"steps"),a=[];if(o>0)throw new yt("Verbose mode is not implemented yet.");if(r!=null)throw new yt("steps mode in testLoop() is not implemented yet");{const l=cu(i,s),c=Be(Rn(0,i));for(let u=0;u<l.length;++u){const h=l[u][0],d=l[u][1],p=so(c,h,d-h),f=zu(e,p),m=t(f);if(u===0)for(let g=0;g<m.length;++g)a.push(mt(0));for(let g=0;g<m.length;++g){const x=m[g];a[g]=J(a[g],A(d-h,x))}}for(let u=0;u<a.length;++u)a[u]=ct(a[u],i)}return a})}getDedupedMetricsNames(){const t=this.metricsNames,e=[];for(let s=0;s<t.length;++s){const o=t[s];let r=o;if(Uf(t,o)>1){const i=Uf(t.slice(0,s),o);r+=`_${i}`}e.push(r)}return e}makeTrainFunction(){return t=>{const e=[],s=t.slice(0,this.inputs.length),o=t.slice(this.inputs.length,this.inputs.length+this.outputs.length),r=t.slice(this.inputs.length+this.outputs.length,this.inputs.length+this.outputs.length*2),i=[],a=()=>{const h=[];for(let m=0;m<this.inputs.length;++m)h.push({key:this.inputs[m],value:s[m]});const d=new $s(h),p=Cr(this.outputs,d,{training:!0});let f;for(let m=0;m<this.lossFunctions.length;++m){const g=this.lossFunctions[m];let x=g(o[m],p[m]);r[m]!=null&&(x=NA(x,r[m]));const b=te(x);e.push(b),m===0?f=x:f=J(f,x)}for(let m=0;m<this.metricsTensors.length;++m){let g;if(this.outputs.length>1&&m<this.outputs.length)g=e[m];else{const x=this.metricsTensors[m][0],b=this.metricsTensors[m][1];g=te(x(o[b],p[b]))}Bn(g),i.push(g)}return f=te(f),this.calculateLosses().forEach(m=>{f=J(f,m)}),f},l=this.collectedTrainableWeights.map(h=>h.read());return[this.optimizer_.minimize(a,!0,l)].concat(i)}}makeTestFunction(){this.testFunction=t=>z(()=>{const e=[];let s;const o=t.slice(0,this.inputs.length),r=t.slice(this.inputs.length,this.inputs.length+this.outputs.length),i=[];for(let c=0;c<this.inputs.length;++c)i.push({key:this.inputs[c],value:o[c]});const a=new $s(i),l=Cr(this.outputs,a);for(let c=0;c<this.lossFunctions.length;++c){const u=this.lossFunctions[c],h=te(u(r[c],l[c]));c===0?s=h:s=J(s,h),e.push(s)}for(let c=0;c<this.metricsTensors.length;++c){const u=this.metricsTensors[c][0],h=this.metricsTensors[c][1],d=te(u(r[h],l[h]));e.push(d)}return e})}async fit(t,e,s={}){if(this.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");this.isTraining=!0;let o,r,i,a,l,c,u,h,d;try{const p=s.batchSize==null?32:s.batchSize;lu(p);const m=await this.standardizeUserData(t,e,s.sampleWeight,s.classWeight,!1,p);o=m[0],r=m[1],d=m[2];let g=!1,x;if(s.validationData!=null&&s.validationData.length>0){if(g=!0,s.validationData.length===2)l=s.validationData[0],c=s.validationData[1];else throw s.validationData.length===3?new yt("validationData including sample weights is not supported yet."):new F(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${s.validationData} is invalid.`);const S=await this.standardizeUserData(l,c,null,null,!0,p);u=S[0],h=S[1],x=u.concat(h)}else if(s.validationSplit!=null&&s.validationSplit>0&&s.validationSplit<1){g=!0;const k=Math.floor(o[0].shape[0]*(1-s.validationSplit)),S=o[0].shape[0];u=br(o,k,S),i=o,o=br(o,0,k),h=br(r,k,S),a=r,r=br(r,0,k),x=u.concat(h)}else s.validationSteps!=null&&(g=!0);const b=o.concat(r).concat(d);this.checkTrainableWeightsConsistency();const w=this.makeTrainFunction(),y=this.getDedupedMetricsNames();let $,v;g?(this.makeTestFunction(),$=this.testFunction,v=y.slice().concat(y.map(k=>"val_"+k))):($=null,x=[],v=y.slice());const N=Tb(s.callbacks,s.yieldEvery);return await this.fitLoop(w,b,y,p,s.epochs,s.verbose,N,$,x,s.shuffle,v,s.initialEpoch,null,null)}finally{this.isTraining=!1,$n(o,t),$n(r,e),$n(i,t),$n(a,e),$n(u,l),$n(h,c),d!=null&&xt(d)}}async fitLoop(t,e,s,o,r,i,a,l,c,u,h,d,p,f){o==null&&(o=32),r==null&&(r=1),u==null&&(u=!0),d==null&&(d=0);let m=!1;if(l!=null&&c!=null&&(m=!0),f!=null&&(m=!0,p==null))throw new F("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");const g=this.checkNumSamples(e,o,p,"steps_per_epoch");let x;g!=null&&(x=Rn(0,g)),i==null&&(i=1);const{callbackList:b,history:w}=Eb(a,i,r,d,g,p,o,m,h);b.setModel(this),this.history=w,await b.onTrainBegin(),this.stopTraining_=!1;for(let y=d;y<r;++y){await b.onEpochBegin(y);const $={};if(p!=null)throw new yt("stepsPerEpoch mode is not implemented yet.");{if(u==="batch")throw new yt("batch shuffling is not implemneted yet");u&&Zu(x);const v=Be(x),N=cu(g,o);for(let T=0;T<N.length;++T){const k={};if(await b.onBatchBegin(T,k),z(()=>{const S=N[T][0],C=N[T][1],D=so(v,S,C-S);k.batch=T,k.size=C-S;const O=zu(e,D),P=t(O);for(let M=0;M<s.length;++M){const V=s[M],H=P[M];k[V]=H,Bn(H)}if(T===N.length-1&&m){const M=this.testLoop(l,c,o);for(let V=0;V<s.length;++V){const H=s[V],W=M[V];Bn(W),$["val_"+H]=W}}}),await b.onBatchEnd(T,k),Nb(k),this.stopTraining_)break}v.dispose()}if(await b.onEpochEnd(y,$),this.stopTraining_)break}return await b.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(t,e){return RA(this,t,e)}async trainOnBatch(t,e){const s=await this.standardizeUserData(t,e),o=s[0],r=s[1],a=this.makeTrainFunction()(o.concat(r)),l=[];for(const c of a){const u=await c.data();l.push(u[0])}return xt(a),$n(s[0],t),$n(s[1],e),Pe(l)}getNamedWeights(t){const e=[],s=t!=null&&t.trainableOnly,o=s?this.trainableWeights:this.weights,r=this.getWeights(s);for(let i=0;i<o.length;++i)s&&!o[i].trainable||e.push({name:o[i].originalName,tensor:r[i]});return e}set stopTraining(t){this.stopTraining_=t}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(t){this.optimizer_!==t&&(this.optimizer_=t,this.isOptimizerOwned=!1)}dispose(){const t=super.dispose();if(t.refCountAfterDispose===0&&this.optimizer!=null&&this.isOptimizerOwned){const e=mf().numTensors;this.optimizer_.dispose(),t.numDisposedVariables+=e-mf().numTensors}return t}getLossIdentifiers(){let t;if(typeof this.loss=="string")t=Zn(this.loss);else if(Array.isArray(this.loss)){for(const e of this.loss)if(typeof e!="string")throw new Error("Serialization of non-string loss is not supported.");t=this.loss.map(e=>Zn(e))}else{const e=Object.keys(this.loss);t={};const s=this.loss;for(const o of e)if(typeof s[o]=="string")t[o]=Zn(s[o]);else throw new Error("Serialization of non-string loss is not supported.")}return t}getMetricIdentifiers(){if(typeof this.metrics=="string"||typeof this.metrics=="function")return[Zn(ba(this.metrics))];if(Array.isArray(this.metrics))return this.metrics.map(t=>Zn(ba(t)));{const t={};for(const e in this.metrics)t[e]=Zn(ba(this.metrics[e]));return t}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(t){if(t.weighted_metrics!=null)throw new Error("Loading weight_metrics is not supported yet.");if(t.loss_weights!=null)throw new Error("Loading loss_weights is not supported yet.");if(t.sample_weight_mode!=null)throw new Error("Loading sample_weight_mode is not supported yet.");const e=Br(t.optimizer_config),s=Tn(e);let o;if(typeof t.loss=="string")o=Ys(t.loss);else if(Array.isArray(t.loss))o=t.loss.map(i=>Ys(i));else if(t.loss!=null){o={};for(const i in t.loss)o[i]=Ys(t.loss[i])}let r;if(Array.isArray(t.metrics))r=t.metrics.map(i=>Ys(i));else if(t.metrics!=null){r={};for(const i in t.metrics)r[i]=Ys(t.metrics[i])}this.compile({loss:o,metrics:r,optimizer:s})}async save(t,e){if(typeof t=="string"){const c=CC(t);if(c.length===0)throw new F(`Cannot find any save handlers for URL '${t}'`);if(c.length>1)throw new F(`Found more than one (${c.length}) save handlers for URL '${t}'`);t=c[0]}if(t.save==null)throw new F("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");const s=await gf(this.getNamedWeights(e)),a={modelTopology:this.toJSON(null,!1),format:PA,generatedBy:`TensorFlow.js tfjs-layers v${Mb}`,convertedBy:null};if((e==null?!1:e.includeOptimizer)&&this.optimizer!=null){a.trainingConfig=this.getTrainingConfig();const c="optimizer",{data:u,specs:h}=await gf(await this.optimizer.getWeights(),c);s.specs.push(...h),s.data=fC([s.data,u])}return this.userDefinedMetadata!=null&&(tm(this.userDefinedMetadata,this.name,!0),a.userDefinedMetadata=this.userDefinedMetadata),a.weightData=s.data,a.weightSpecs=s.specs,t.save(a)}setUserDefinedMetadata(t){tm(t,this.name),this.userDefinedMetadata=t}getUserDefinedMetadata(){return this.userDefinedMetadata}}Uo.className="Model";X(Uo);class Wb extends Uo{}Wb.className="Functional";X(Wb);async function c6(n,t){"modelTopology"in n||(n={modelTopology:n}),n=n;let e=n.modelTopology;e.model_config!=null&&(e=e.model_config);const s=Br(e),o=Tn(s,t);if(n.weightsManifest!=null){const r=await VN(n.weightsManifest,n.pathPrefix,o.weights.map(a=>a.originalName)),i={};for(const a of o.weights)i[a.originalName]=r[a.originalName];o.loadWeights(i),xt(r)}return o}async function u6(n,t){if(t==null&&(t={}),typeof n=="string"){const e=IC(n,t);if(e.length===0)e.push(qN(n,t));else if(e.length>1)throw new F(`Found more than one (${e.length}) load handlers for URL '${n}'`);n=e[0]}return zA(n,void 0,t)}async function zA(n,t,e){if(e==null&&(e={}),n.load==null)throw new F("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const s=await n.load();let o=s.modelTopology;o.model_config!=null&&(o=o.model_config);const r=e.strict==null?!0:e.strict,i=s.weightData!=null&&s.weightSpecs!=null&&r,a=Tn(Br(o),t,i),l=s.trainingConfig;if(l!=null&&a.loadTrainingConfig(l),s.userDefinedMetadata!=null&&a.setUserDefinedMetadata(s.userDefinedMetadata),s.weightData!=null){if(s.weightSpecs==null)throw new F("LayersModel artifacts contains weight data, but not weight specs. Therefore loading of weights cannot proceed.");const{modelWeights:c,optimizerWeights:u}=BA(s.weightData,s.weightSpecs);a.loadWeights(c,r),a.optimizer!=null&&u.length>0&&await a.optimizer.setWeights(u),xt(c),xt(u.map(h=>h.tensor))}return a}function BA(n,t){const e=Ng(n,t),s={},o=[];return t.forEach(r=>{r.group==="optimizer"?o.push({name:r.name,tensor:e[r.name]}):s[r.name]=e[r.name]}),{modelWeights:s,optimizerWeights:o}}class Vr extends Uo{constructor(t){if(super({inputs:[],outputs:[]}),t=t||{},this.trainable=!0,this.built=!1,this.name=t.name!=null?t.name:Ac("sequential_"),t.layers!=null)for(const e of t.layers)this.add(e)}checkShape(t){if(t.inboundNodes[0].outputTensors[0].shape.some(s=>s<0))throw new F(`Negative dimension size caused by adding layer ${t.name} with input shape [${t.inboundNodes[0].inputTensors[0].shape}]`)}add(t){const e=t instanceof Vr||t instanceof Uo;let s;if(e){if(s=t,s.outputs.length!==1)throw new F("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");if(s.inputs.length!==1)throw new F("All layers in a Sequential model should have a single input tensor. For multi-input layers, use the functional API.")}if(this.outputs.length===0){if(t.inboundNodes.length===0){if(t.batchInputShape==null)throw new F("The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.");const o=BR({batchShape:t.batchInputShape,dtype:t.dtype,name:t.name+"_input"});t.apply(o)}if(e)this.outputs=s.outputs,this.inputs=s.inputs;else{if(t.inboundNodes.length!==1)throw new F(`A layer added to a Sequential model must not already be connected somewhere else. LayersModel received layer ${t.name} which has ${t.inboundNodes.length} pre-existing inbound connections.`);if(t.inboundNodes[0].outputTensors.length!==1)throw new F("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(t),this.outputs=[t.inboundNodes[0].outputTensors[0]],this.inputs=Cb(this.outputs[0])}this.inboundNodes=[],new Fc({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:bo(null,this.inputs.length),outputMasks:[null],inputShapes:this.inputs.map(o=>o.shape),outputShapes:this.outputs[0].shape})}else{const o=t.apply(this.outputs[0]);if(Array.isArray(o))throw new TypeError("All layers in a Sequential model should have a single output tensor. For multi-output layers, use the functional API.");this.checkShape(t),this.outputs=[o],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}this.layers.push(t),this.built=!1}pop(){if(this.layers.length===0)throw new TypeError("There are no layers in the model.");if(this.layers.pop(),this.layers.length===0)this.outputs=[],this.inboundNodes=[],this.outboundNodes=[];else{const t=this.layers.length-1;this.layers[t].outboundNodes=[],this.outputs=[this.layers[t].output],this.inboundNodes[0].outputTensors=this.outputs,this.inboundNodes[0].outputShapes=[this.outputs[0].shape]}}call(t,e){return this.model==null&&this.build(),this.model.call(t,e)}build(t){if(Nt(t),this.inputs.length===0||this.outputs.length===0)throw new TypeError("Sequential model cannot be built: model is empty. Add some layers first.");this.model=new Uo({inputs:this.inputs,outputs:this.outputs[0],name:this.name+"_model"}),this.model.trainable=this.trainable,this.supportsMasking=this.model.supportsMasking,this.inputLayers=this.model.inputLayers,this.inputLayersNodeIndices=this.model.inputLayersNodeIndices,this.inputLayersTensorIndices=this.model.inputLayersTensorIndices,this.outputLayers=this.model.outputLayers,this.outputLayersNodeIndices=this.model.outputLayersNodeIndices,this.outputLayersTensorIndices=this.model.outputLayersTensorIndices,this.nodesByDepth=this.model.nodesByDepth,this.containerNodes=this.model.containerNodes,this.outputNames=this.model.outputNames,this.inputNames=this.model.inputNames,this.built=!0}countParams(){return this.built||this.build(),super.countParams()}summary(t,e,s=console.log){this.built||this.build(),super.summary(t,e,s)}setWeights(t){this.model==null&&this.build(),this.model.setWeights(t)}evaluate(t,e,s={}){if(!this.built)throw new dn("The model needs to be compiled before being used.");return this.model.evaluate(t,e,s)}async evaluateDataset(t,e){if(!this.built)throw new dn("The model needs to be compiled before being used.");return this.model.evaluateDataset(t,e)}predict(t,e={}){return this.model==null&&this.build(),this.model.predict(t,e)}predictOnBatch(t){return this.model==null&&this.build(),this.model.predictOnBatch(t)}compile(t){this.build(),this.model.compile(t),this.optimizer_=this.model.optimizer,this.isOptimizerOwned=this.model.isOptimizerOwned,this.loss=this.model.loss,this.metrics=this.model.metrics,this.metricsTensors=this.model.metricsTensors,this.metricsNames=this.model.metricsNames}get optimizer(){return this.model==null?void 0:this.model.optimizer}set optimizer(t){this.model.optimizer=t}async fit(t,e,s={}){if(!this.built)throw new dn("The model needs to be compiled before being used.");return this.model.fit(t,e,s)}async fitDataset(t,e){if(!this.built)throw new dn("The model needs to be compiled before being used.");return this.model.fitDataset(t,e)}async trainOnBatch(t,e){return this.model.trainOnBatch(t,e)}static fromConfig(t,e,s={},o=!1){let r,i={};if(e instanceof Array){if(e[0].className==null||e[0].className==="Merge")throw new F("Legacy serialization format not supported yet.");r=e}else I(e.layers!=null,()=>"When the config data for a Sequential model is not an Array, it must be an Object that contains the 'layers' field."),r=e.layers,delete e.layers,i=e;const a=new t(i);if(!(a instanceof Vr))throw new yt(`Sequential.fromConfig called on non-Sequential input: ${a}`);for(const l of r){const u=Tn(l,void 0,o);o&&u.setFastWeightInitDuringBuild(!0),a.add(u)}return a}set stopTraining(t){if(this.model==null)throw new F("Cannot set the stopTraining property of a sequential model before it is compiled.");this.model.stopTraining=t}get stopTraining(){if(this.model==null)throw new F("Cannot get the stopTraining property of a sequential model before it is compiled.");return this.model.stopTraining}getConfig(){const t=[];for(const e of this.layers){const s={};s.className=e.getClassName(),s.config=e.getConfig(),t.push(s)}return{name:this.name,layers:t}}}Vr.className="Sequential";X(Vr);let Ee=class extends No{getConfig(){return{}}};class Ub extends Ee{apply(t,e=1){return vR(t,e)}}Ub.className="elu";X(Ub);class Gb extends Ee{apply(t){return Qg(t)}}Gb.className="selu";X(Gb);class Hb extends Ee{apply(t){return ds(t)}}Hb.className="relu";X(Hb);class qb extends Ee{apply(t){return z(()=>jo(6,ds(t)))}}qb.className="relu6";X(qb);class jb extends Ee{apply(t){return t}}jb.className="linear";X(jb);class Kb extends Ee{apply(t){return er(t)}}Kb.className="sigmoid";X(Kb);class Xb extends Ee{apply(t){return SR(t)}}Xb.className="hardSigmoid";X(Xb);class Yb extends Ee{apply(t){return ji(t)}}Yb.className="softplus";X(Yb);class Zb extends Ee{apply(t){return kR(t)}}Zb.className="softsign";X(Zb);class Jb extends Ee{apply(t){return wc(t)}}Jb.className="tanh";X(Jb);let Sp=class extends Ee{apply(t,e=-1){return Sd(t,e)}};Sp.className="softmax";X(Sp);class Qb extends Ee{apply(t,e=-1){return jg(t,e)}}Qb.className="logSoftmax";X(Qb);class t0 extends Ee{apply(t){return z(()=>z(()=>{const e=Math.sqrt(2),s=A(.5,J(1,Ug(ct(t,e))));return A(t,s)}))}}t0.className="gelu";X(t0);class e0 extends Ee{apply(t){return z(()=>A(.5,A(t,J(1,wc(A(Ne(ct(2,Math.PI)),J(t,A(.044715,fo(t,3)))))))))}}e0.className="gelu_new";X(e0);class n0 extends Ee{apply(t){return z(()=>A(t,wc(ji(t))))}}n0.className="mish";X(n0);class s0 extends Ee{apply(t,e=1){return z(()=>A(er(A(t,e)),t))}}s0.className="swish";X(s0);function Ds(n){return n.getClassName()}function uu(n,t={}){return ta(n,tn.getMap().classNameMap,t,"activation")}function Fs(n){if(n==null){const t={};return t.className="linear",t.config={},uu(t)}if(typeof n=="string"){const t={};return t.className=n,t.config={},uu(t)}else return n instanceof Ee?n:uu(n)}function Np(n){if(n!=null&&typeof n!="object")throw new Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${n}`)}class o0 extends No{}class Lc extends o0{constructor(t){super(),Np(t),this.l1=t==null||t.l1==null?.01:t.l1,this.l2=t==null||t.l2==null?.01:t.l2,this.hasL1=this.l1!==0,this.hasL2=this.l2!==0}apply(t){return z(()=>{let e=pe([1]);return this.hasL1&&(e=J(e,it(A(this.l1,he(t))))),this.hasL2&&(e=J(e,it(A(this.l2,na(t))))),L(e,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(t,e){return new t({l1:e.l1,l2:e.l2})}}Lc.className="L1L2";X(Lc);function p6(n){return Np(n),new Lc({l1:n!=null?n.l1:null,l2:0})}function f6(n){return Np(n),new Lc({l2:n!=null?n.l2:null,l1:0})}const im={l1l2:"L1L2"};function Mt(n){return dp(n)}function am(n,t={}){return ta(n,tn.getMap().classNameMap,t,"regularizer")}function qt(n){if(n==null)return null;if(typeof n=="string"){const e={className:n in im?im[n]:n,config:{}};return am(e)}else return n instanceof o0?n:am(n)}class r0 extends It{constructor(t){super(t??{}),this.supportsMasking=!0,t!=null&&(this.maxValue=t.maxValue)}call(t,e){t=ft(t);let s=ds(t);return this.maxValue!=null&&(s=on(s,0,this.maxValue)),s}computeOutputShape(t){return t}getConfig(){const t={maxValue:this.maxValue},e=super.getConfig();return Object.assign(t,e),t}}r0.className="ReLU";X(r0);class i0 extends It{constructor(t){super(t??{}),this.DEFAULT_ALPHA=.3,t==null&&(t={}),this.alpha=t.alpha==null?this.DEFAULT_ALPHA:t.alpha}call(t,e){const s=ft(t);return fd(s,this.alpha)}computeOutputShape(t){return t}getConfig(){const t={alpha:this.alpha},e=super.getConfig();return Object.assign(t,e),t}}i0.className="LeakyReLU";X(i0);class a0 extends It{constructor(t){if(super(t??{}),this.DEFAULT_ALPHA_INITIALIZER="zeros",t==null&&(t={}),this.supportsMasking=!0,this.alphaInitializer=Gt(t.alphaInitializer||this.DEFAULT_ALPHA_INITIALIZER),this.alphaRegularizer=qt(t.alphaRegularizer),this.alphaConstraint=ce(t.alphaConstraint),t.sharedAxes==null)this.sharedAxes=null;else if(Array.isArray(t.sharedAxes))this.sharedAxes=t.sharedAxes;else if(typeof t.sharedAxes=="number")this.sharedAxes=[t.sharedAxes];else throw new F(`Expected sharedAxes to be a number or an array of numbers, but got ${t.sharedAxes}`)}build(t){t=Nt(t);const e=t.slice(1);if(this.sharedAxes!=null)for(const o of this.sharedAxes)e[o-1]=1;this.alpha=this.addWeight("alpha",e,"float32",this.alphaInitializer,this.alphaRegularizer,!0,this.alphaConstraint);const s={};if(this.sharedAxes!=null)for(let o=1;o<t.length;++o)s[o]=t[o];this.inputSpec=[new ae({ndim:t.length,axes:s})],this.built=!0}call(t,e){return t=ft(t),Cd(t,this.alpha.read())}getConfig(){const t={alphaInitializer:Kt(this.alphaInitializer),alphaRegularizer:Mt(this.alphaRegularizer),alphaConstraint:le(this.alphaConstraint),sharedAxes:this.sharedAxes},e=super.getConfig();return Object.assign(t,e),t}}a0.className="PReLU";X(a0);let l0=class extends It{constructor(t){if(super(t??{}),this.DEFAULT_ALPHA=1,t==null&&(t={}),t.alpha!=null&&t.alpha!==this.DEFAULT_ALPHA)throw new yt(`Non-default alpha value (${t.alpha}) is not supported by the ELU layer yet.`);this.alpha=t.alpha==null?this.DEFAULT_ALPHA:t.alpha}call(t,e){const s=ft(t);return Cc(s)}computeOutputShape(t){return t}getConfig(){const t={alpha:this.alpha},e=super.getConfig();return Object.assign(t,e),t}};l0.className="ELU";X(l0);class c0 extends It{constructor(t){super(t??{}),this.DEFAULT_THETA=1,t==null&&(t={}),this.theta=t.theta==null?this.DEFAULT_THETA:t.theta}call(t,e){const s=ft(t);return A(s,st(Ze(s,this.theta),"float32"))}computeOutputShape(t){return t}getConfig(){const t={theta:this.theta},e=super.getConfig();return Object.assign(t,e),t}}c0.className="ThresholdedReLU";X(c0);class u0 extends It{constructor(t){super(t??{}),this.DEFAULT_AXIS=1,t==null&&(t={}),this.softmax=new Sp().apply,this.axis=t.axis==null?this.DEFAULT_AXIS:t.axis}call(t,e){return z(()=>{let s=ft(t);const o=e.mask;if(o!=null){const r=A(rt(hs(s.shape),st(o,s.dtype)),mt(-1e9));s=J(s,r)}return this.axis instanceof Array?this.axis.length>1?rn(rt(s,gd(s,this.axis,!0))):this.softmax(s,this.axis[0]):this.softmax(s,this.axis)})}computeOutputShape(t){return t}getConfig(){const t={axis:this.axis},e=super.getConfig();return Object.assign(t,e),t}}u0.className="Softmax";X(u0);function Go(n,t,e){if(typeof n=="number")return bo(n,t);if(n.length!==t)throw new F(`The ${e} argument must be an integer or tuple of ${t} integers. Received: ${n.length} elements.`);for(let s=0;s<t;++s){const o=n[s];if(!wR(o))throw new F(`The ${e} argument must be an integer or tuple of ${t} integers. Received: ${JSON.stringify(n)} including a non-integer number ${o}`)}return n}function En(n,t,e,s,o=1){if(n==null)return n;const r=t+(t-1)*(o-1);let i;return e==="same"?i=n:i=n-r+1,Math.floor((i+s-1)/s)}function Pn(n,t,e,s){if(n==null)return null;if(s==="valid")n=n*t+As([e-t,0]);else if(s==="same")n=n*t;else throw new F(`Unsupport padding mode: ${s}.`);return n}function Tp(n,t){return z(()=>(se(t),t==="channelsFirst"?vt(n,[0,2,3,1]):n))}function h0(n,t){return z(()=>(se(t),t==="channelsFirst"?vt(n,[0,2,3,4,1]):n))}function VA(n,t,e,s=1,o="valid",r,i=1){return z(()=>{if(r==null&&(r=An()),se(r),n.shape.length!==3)throw new F(`The input of a conv1dWithBias operation should be 3, but is ${n.shape.length} instead.`);if(t.shape.length!==3)throw new F(`The kernel for a conv1dWithBias operation should be 3, but is ${t.shape.length} instead`);if(e!=null&&e.shape.length!==1)throw new F(`The bias for a conv1dWithBias operation should be 1, but is ${e.shape.length} instead`);if(r==="channelsFirst"&&(n=vt(n,[0,2,1])),o==="causal")throw new yt("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");let a=Pg(n,t,s,o==="same"?"same":"valid","NWC",i);return e!=null&&(a=_n(a,e)),a})}function lm(n,t,e,s=[1,1],o="valid",r,i,a=null){return z(()=>{if(r==null&&(r=An()),se(r),n.rank!==3&&n.rank!==4)throw new F(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${n.rank}.`);if(t.rank!==3&&t.rank!==4)throw new F(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${n.rank}.`);let l=Tp(n,r);if(o==="causal")throw new yt("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");return l=e2({x:l,filter:t,strides:s,pad:o==="same"?"same":"valid",dilations:i,dataFormat:"NHWC",bias:e,activation:a}),r==="channelsFirst"&&(l=vt(l,[0,3,1,2])),l})}function WA(n,t,e,s=[1,1,1],o="valid",r,i){return z(()=>{if(r==null&&(r=An()),se(r),n.rank!==4&&n.rank!==5)throw new F(`conv3dWithBias expects input to be of rank 4 or 5, but received ${n.rank}.`);if(t.rank!==4&&t.rank!==5)throw new F(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${n.rank}.`);let a=h0(n,r);if(o==="causal")throw new yt("The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.");return a=QI(a,t,s,o==="same"?"same":"valid","NDHWC",i),e!=null&&(a=_n(a,e)),r==="channelsFirst"&&(a=vt(a,[0,4,1,2,3])),a})}class Mc extends It{constructor(t,e){if(super(e),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",Mc.verifyArgs(e),this.rank=t,fe(this.rank,"rank"),this.rank!==1&&this.rank!==2&&this.rank!==3)throw new yt(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=Go(e.kernelSize,t,"kernelSize"),this.strides=Go(e.strides==null?1:e.strides,t,"strides"),this.padding=e.padding==null?"valid":e.padding,an(this.padding),this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,se(this.dataFormat),this.activation=Fs(e.activation),this.useBias=e.useBias==null?!0:e.useBias,this.biasInitializer=Gt(e.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=ce(e.biasConstraint),this.biasRegularizer=qt(e.biasRegularizer),this.activityRegularizer=qt(e.activityRegularizer),this.dilationRate=Go(e.dilationRate==null?1:e.dilationRate,t,"dilationRate"),this.rank===1&&Array.isArray(this.dilationRate)&&this.dilationRate.length!==1)throw new F(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(this.rank===2){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==2)throw new F(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(this.rank===3){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==3)throw new F(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(t){if(Mn("kernelSize"in t,"required key 'kernelSize' not in config"),typeof t.kernelSize!="number"&&!pp(t.kernelSize,"number",1,3))throw new F(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(t.kernelSize)}.`)}getConfig(){const t={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:Ds(this.activation),useBias:this.useBias,biasInitializer:Kt(this.biasInitializer),biasRegularizer:Mt(this.biasRegularizer),activityRegularizer:Mt(this.activityRegularizer),biasConstraint:le(this.biasConstraint)},e=super.getConfig();return Object.assign(t,e),t}}class rr extends Mc{constructor(t,e){super(t,e),this.kernel=null,rr.verifyArgs(e),this.filters=e.filters,fe(this.filters,"filters"),this.kernelInitializer=Gt(e.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=ce(e.kernelConstraint),this.kernelRegularizer=qt(e.kernelRegularizer)}build(t){t=Nt(t);const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new F(`The channel dimension of the input should be defined. Found ${t[e]}`);const s=t[e],o=this.kernelSize.concat([s,this.filters]);this.kernel=this.addWeight("kernel",o,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[e]:s}}],this.built=!0}call(t,e){return z(()=>{t=ft(t);let s;const o=this.bias==null?null:this.bias.read(),r=ab(this.activation.getClassName());if(r!=null&&this.rank===2)s=lm(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate,r);else{if(this.rank===1)s=VA(t,this.kernel.read(),o,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(this.rank===2)s=lm(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(this.rank===3)s=WA(t,this.kernel.read(),o,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new yt("convolutions greater than 3D are not implemented yet.");this.activation!=null&&(s=this.activation.apply(s))}return s})}computeOutputShape(t){t=Nt(t);const e=[],s=this.dataFormat==="channelsLast"?t.slice(1,t.length-1):t.slice(2);for(let r=0;r<s.length;++r){const i=En(s[r],this.kernelSize[r],this.padding,this.strides[r],typeof this.dilationRate=="number"?this.dilationRate:this.dilationRate[r]);e.push(i)}let o=[t[0]];return this.dataFormat==="channelsLast"?(o=o.concat(e),o.push(this.filters)):(o.push(this.filters),o=o.concat(e)),o}getConfig(){const t={filters:this.filters,kernelInitializer:Kt(this.kernelInitializer),kernelRegularizer:Mt(this.kernelRegularizer),kernelConstraint:le(this.kernelConstraint)},e=super.getConfig();return Object.assign(t,e),t}static verifyArgs(t){if(!("filters"in t)||typeof t.filters!="number"||t.filters<1)throw new F(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(t.filters)}`)}}class ia extends rr{constructor(t){super(2,t),ia.verifyArgs(t)}getConfig(){const t=super.getConfig();return delete t.rank,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!pp(t.kernelSize,"number",1,2))throw new F(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(t.kernelSize)}.`)}}ia.className="Conv2D";X(ia);class aa extends rr{constructor(t){super(3,t),aa.verifyArgs(t)}getConfig(){const t=super.getConfig();return delete t.rank,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!(Array.isArray(t.kernelSize)&&(t.kernelSize.length===1||t.kernelSize.length===3)))throw new F(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(t.kernelSize)}.`)}}aa.className="Conv3D";X(aa);class d0 extends ia{constructor(t){if(super(t),this.inputSpec=[new ae({ndim:4})],this.padding!=="same"&&this.padding!=="valid")throw new F(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(t){if(t=Nt(t),t.length!==4)throw new F("Input should have rank 4; Received input shape: "+JSON.stringify(t));const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new F("The channel dimension of the inputs should be defined. Found `None`.");const s=t[e],o=this.kernelSize.concat([this.filters,s]);this.kernel=this.addWeight("kernel",o,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new ae({ndim:4,axes:{[e]:s}})],this.built=!0}call(t,e){return z(()=>{let s=ft(t);if(s.shape.length!==4)throw new F(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${s.shape.length}`);const o=s.shape,r=o[0];let i,a;this.dataFormat==="channelsFirst"?(i=2,a=3):(i=1,a=2);const l=o[i],c=o[a],u=this.kernelSize[0],h=this.kernelSize[1],d=this.strides[0],p=this.strides[1],f=Pn(l,d,u,this.padding),m=Pn(c,p,h,this.padding),g=[r,f,m,this.filters];this.dataFormat!=="channelsLast"&&(s=vt(s,[0,2,3,1]));let x=zg(s,this.kernel.read(),g,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(x=vt(x,[0,3,1,2])),this.bias!=null&&(x=_n(x,this.bias.read(),this.dataFormat)),this.activation!=null&&(x=this.activation.apply(x)),x})}computeOutputShape(t){t=Nt(t);const e=t.slice();let s,o,r;this.dataFormat==="channelsFirst"?(s=1,o=2,r=3):(s=3,o=1,r=2);const i=this.kernelSize[0],a=this.kernelSize[1],l=this.strides[0],c=this.strides[1];return e[s]=this.filters,e[o]=Pn(e[o],l,i,this.padding),e[r]=Pn(e[r],c,a,this.padding),e}getConfig(){const t=super.getConfig();return delete t.dilationRate,t}}d0.className="Conv2DTranspose";X(d0);class p0 extends aa{constructor(t){if(super(t),this.inputSpec=[new ae({ndim:5})],this.padding!=="same"&&this.padding!=="valid")throw new F(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(t){if(t=Nt(t),t.length!==5)throw new F("Input should have rank 5; Received input shape: "+JSON.stringify(t));const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null)throw new F("The channel dimension of the inputs should be defined. Found `None`.");const s=t[e],o=this.kernelSize.concat([this.filters,s]);this.kernel=this.addWeight("kernel",o,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new ae({ndim:5,axes:{[e]:s}})],this.built=!0}call(t,e){return z(()=>{let s=ft(t);if(s.shape.length!==5)throw new F(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${s.shape.length}`);const o=s.shape,r=o[0];let i,a,l;this.dataFormat==="channelsFirst"?(l=2,i=3,a=4):(l=1,i=2,a=3);const c=o[l],u=o[i],h=o[a],d=this.kernelSize[0],p=this.kernelSize[1],f=this.kernelSize[2],m=this.strides[0],g=this.strides[1],x=this.strides[2],b=Pn(c,m,d,this.padding),w=Pn(u,g,p,this.padding),y=Pn(h,x,f,this.padding),$=[r,b,w,y,this.filters];this.dataFormat!=="channelsLast"&&(s=vt(s,[0,2,3,4,1]));let v=nv(s,this.kernel.read(),$,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(v=vt(v,[0,4,1,2,3])),this.bias!==null&&(v=_n(v,this.bias.read(),this.dataFormat)),this.activation!==null&&(v=this.activation.apply(v)),v})}computeOutputShape(t){t=Nt(t);const e=t.slice();let s,o,r,i;this.dataFormat==="channelsFirst"?(s=1,o=2,r=3,i=4):(s=4,o=1,r=2,i=3);const a=this.kernelSize[0],l=this.kernelSize[1],c=this.kernelSize[2],u=this.strides[0],h=this.strides[1],d=this.strides[2];return e[s]=this.filters,e[o]=Pn(e[o],u,a,this.padding),e[r]=Pn(e[r],h,l,this.padding),e[i]=Pn(e[i],d,c,this.padding),e}getConfig(){const t=super.getConfig();return delete t.dilationRate,t}}p0.className="Conv3DTranspose";X(p0);class f0 extends rr{constructor(t,e){if(super(t,e),this.DEFAULT_DEPTHWISE_INITIALIZER="glorotUniform",this.DEFAULT_POINTWISE_INITIALIZER="glorotUniform",this.depthwiseKernel=null,this.pointwiseKernel=null,e.filters==null)throw new F("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(e.kernelInitializer!=null||e.kernelRegularizer!=null||e.kernelConstraint!=null)throw new F("Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.");if(e.padding!=null&&e.padding!=="same"&&e.padding!=="valid")throw new F(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(e.padding)}`);this.depthMultiplier=e.depthMultiplier==null?1:e.depthMultiplier,this.depthwiseInitializer=Gt(e.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=qt(e.depthwiseRegularizer),this.depthwiseConstraint=ce(e.depthwiseConstraint),this.pointwiseInitializer=Gt(e.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=qt(e.pointwiseRegularizer),this.pointwiseConstraint=ce(e.pointwiseConstraint)}build(t){if(t=Nt(t),t.length<this.rank+2)throw new F(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(t)}`);const e=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[e]==null||t[e]<0)throw new F(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(t[e])}`);const s=t[e],o=this.kernelSize.concat([s,this.depthMultiplier]),r=[];for(let a=0;a<this.rank;++a)r.push(1);r.push(s*this.depthMultiplier,this.filters);const i=!0;this.depthwiseKernel=this.addWeight("depthwise_kernel",o,"float32",this.depthwiseInitializer,this.depthwiseRegularizer,i,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight("pointwise_kernel",r,"float32",this.pointwiseInitializer,this.pointwiseRegularizer,i,this.pointwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,i,this.biasConstraint):this.bias=null,this.inputSpec=[new ae({ndim:this.rank+2,axes:{[e]:s}})],this.built=!0}call(t,e){return z(()=>{t=ft(t);let s;if(this.rank===1)throw new yt("1D separable convolution is not implemented yet.");return this.rank===2&&(this.dataFormat==="channelsFirst"&&(t=vt(t,[0,2,3,1])),s=tx(t,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,"NHWC")),this.useBias&&(s=_n(s,this.bias.read(),this.dataFormat)),this.activation!=null&&(s=this.activation.apply(s)),this.dataFormat==="channelsFirst"&&(s=vt(s,[0,3,1,2])),s})}getConfig(){const t=super.getConfig();return delete t.rank,delete t.kernelInitializer,delete t.kernelRegularizer,delete t.kernelConstraint,t.depthwiseInitializer=Kt(this.depthwiseInitializer),t.pointwiseInitializer=Kt(this.pointwiseInitializer),t.depthwiseRegularizer=Mt(this.depthwiseRegularizer),t.pointwiseRegularizer=Mt(this.pointwiseRegularizer),t.depthwiseConstraint=le(this.depthwiseConstraint),t.pointwiseConstraint=le(this.pointwiseConstraint),t}}f0.className="SeparableConv";class m0 extends f0{constructor(t){super(2,t)}}m0.className="SeparableConv2D";X(m0);class Pc extends rr{constructor(t){super(1,t),Pc.verifyArgs(t),this.inputSpec=[{ndim:3}]}getConfig(){const t=super.getConfig();return delete t.rank,delete t.dataFormat,t}static verifyArgs(t){if(typeof t.kernelSize!="number"&&!pp(t.kernelSize,"number",1,1))throw new F(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(t.kernelSize)}.`)}}Pc.className="Conv1D";X(Pc);class g0 extends It{constructor(t){super(t),typeof t.cropping=="number"?this.cropping=[[t.cropping,t.cropping],[t.cropping,t.cropping]]:typeof t.cropping[0]=="number"?this.cropping=[[t.cropping[0],t.cropping[0]],[t.cropping[1],t.cropping[1]]]:this.cropping=t.cropping,this.dataFormat=t.dataFormat===void 0?"channelsLast":t.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(t){return this.dataFormat==="channelsFirst"?[t[0],t[1],t[2]-this.cropping[0][0]-this.cropping[0][1],t[3]-this.cropping[1][0]-this.cropping[1][1]]:[t[0],t[1]-this.cropping[0][0]-this.cropping[0][1],t[2]-this.cropping[1][0]-this.cropping[1][1],t[3]]}call(t,e){return z(()=>{if(t=ft(t),this.dataFormat==="channelsLast"){const s=xa(t,this.cropping[0][0],t.shape[1]-this.cropping[0][0]-this.cropping[0][1],2);return xa(s,this.cropping[1][0],t.shape[2]-this.cropping[1][1]-this.cropping[1][0],3)}else{const s=xa(t,this.cropping[0][0],t.shape[2]-this.cropping[0][0]-this.cropping[0][1],3);return xa(s,this.cropping[1][0],t.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)}})}getConfig(){const t={cropping:this.cropping,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}g0.className="Cropping2D";X(g0);class x0 extends It{constructor(t){super(t),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=t.size==null?this.DEFAULT_SIZE:t.size,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,se(this.dataFormat),this.interpolation=t.interpolation==null?"nearest":t.interpolation,xR(this.interpolation)}computeOutputShape(t){if(this.dataFormat==="channelsFirst"){const e=t[2]==null?null:this.size[0]*t[2],s=t[3]==null?null:this.size[1]*t[3];return[t[0],t[1],e,s]}else{const e=t[1]==null?null:this.size[0]*t[1],s=t[2]==null?null:this.size[1]*t[2];return[t[0],e,s,t[3]]}}call(t,e){return z(()=>{let s=ft(t);const o=s.shape;if(this.dataFormat==="channelsFirst"){s=vt(s,[0,2,3,1]);const r=this.size[0]*o[2],i=this.size[1]*o[3],a=this.interpolation==="nearest"?ts.resizeNearestNeighbor(s,[r,i]):ts.resizeBilinear(s,[r,i]);return vt(a,[0,3,1,2])}else{const r=this.size[0]*o[1],i=this.size[1]*o[2];return this.interpolation==="nearest"?ts.resizeNearestNeighbor(s,[r,i]):ts.resizeBilinear(s,[r,i])}})}getConfig(){const t={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation},e=super.getConfig();return Object.assign(t,e),t}}x0.className="UpSampling2D";X(x0);function UA(n,t,e=[1,1],s="valid",o,r){return z(()=>{o==null&&(o=An()),se(o);let i=Tp(n,o);if(n.rank!==4)throw new F(`Input for depthwiseConv2d is required to be 4-D, but is instead ${n.rank}-D`);if(t.rank!==4)throw new F(`depthwiseKernel is required to be 4-D, but is instead ${t.rank}-D`);return i=ud(i,t,e,s==="same"?"same":"valid","NHWC",r),o==="channelsFirst"&&(i=vt(i,[0,3,1,2])),i})}class b0 extends Mc{constructor(t){super(2,t),this.depthwiseKernel=null,this.depthMultiplier=t.depthMultiplier==null?1:t.depthMultiplier,this.depthwiseInitializer=Gt(t.depthwiseInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.depthwiseConstraint=ce(t.depthwiseConstraint),this.depthwiseRegularizer=qt(t.depthwiseRegularizer)}build(t){if(t=Nt(t),t.length<4)throw new F(`Inputs to DepthwiseConv2D should have rank 4. Received input shape: ${JSON.stringify(t)}.`);const e=this.dataFormat==="channelsFirst"?1:3;if(t[e]==null||t[e]<0)throw new F(`The channel dimension of the inputs to DepthwiseConv2D should be defined, but is not (${t[e]}).`);const s=t[e],o=[this.kernelSize[0],this.kernelSize[1],s,this.depthMultiplier];this.depthwiseKernel=this.addWeight("depthwise_kernel",o,null,this.depthwiseInitializer,this.depthwiseRegularizer,!0,this.depthwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[s*this.depthMultiplier],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return z(()=>{t=ft(t);let s=UA(t,this.depthwiseKernel.read(),this.strides,this.padding,this.dataFormat,null);return this.useBias&&(s=_n(s,this.bias.read(),this.dataFormat)),this.activation!=null&&(s=this.activation.apply(s)),s})}computeOutputShape(t){t=Nt(t);const e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2],o=this.dataFormat==="channelsFirst"?t[1]*this.depthMultiplier:t[3]*this.depthMultiplier,r=En(e,this.kernelSize[0],this.padding,this.strides[0]),i=En(s,this.kernelSize[1],this.padding,this.strides[1]);return this.dataFormat==="channelsFirst"?[t[0],o,r,i]:[t[0],r,i,o]}getConfig(){const t=super.getConfig();return t.depthMultiplier=this.depthMultiplier,t.depthwiseInitializer=Kt(this.depthwiseInitializer),t.depthwiseRegularizer=Mt(this.depthwiseRegularizer),t.depthwiseConstraint=le(this.depthwiseRegularizer),t}}b0.className="DepthwiseConv2D";X(b0);function y0(n,t,e,s){if(Array.isArray(n)){if(t!=null||e!=null)throw new F("When inputs is an array, neither initialState or constants should be provided");s!=null&&(e=n.slice(n.length-s,n.length),n=n.slice(0,n.length-s)),n.length>1&&(t=n.slice(1,n.length)),n=n[0]}function o(r){return r==null||Array.isArray(r)?r:[r]}return t=o(t),e=o(e),{inputs:n,initialState:t,constants:e}}function w0(n,t,e,s=!1,o,r,i=!1,a=!1){return z(()=>{const l=t.shape.length;if(l<3)throw new F(`Input should be at least 3D, but is ${l}D.`);const c=[1,0].concat(Rn(2,l));t=vt(t,c),i&&console.warn("Backend rnn(): the unroll = true option is not applicable to the imperative deeplearn.js backend."),o!=null&&(o=st(st(o,"bool"),"float32"),o.rank===l-1&&(o=Me(o,-1)),o=vt(o,c)),s&&(t=go(t,0),o!=null&&(o=go(o,0)));const u=[];let h,d=e;const p=t.shape[0],f=xo(t);let m;o!=null&&(m=xo(o));for(let x=0;x<p;++x){const b=f[x],w=z(()=>n(b,d));if(o==null)h=w[0],d=w[1];else{const y=z(()=>{const $=m[x],v=rt(gn($),$),N=J(A(w[0],$),A(d[0],v)),T=d.map((k,S)=>J(A(w[1][S],$),A(k,v)));return{output:N,newStates:T}});h=y.output,d=y.newStates}a&&u.push(h)}let g;return a&&(g=Hn(u,1)),[h,g,d]})}class Ws extends It{constructor(t){super(t);let e;if(t.cell==null)throw new F("cell property is missing for the constructor of RNN.");if(Array.isArray(t.cell)?e=new Ap({cells:t.cell}):e=t.cell,e.stateSize==null)throw new F("The RNN cell should have an attribute `stateSize` (tuple of integers, one integer per RNN state).");this.cell=e,this.returnSequences=t.returnSequences==null?!1:t.returnSequences,this.returnState=t.returnState==null?!1:t.returnState,this.goBackwards=t.goBackwards==null?!1:t.goBackwards,this._stateful=t.stateful==null?!1:t.stateful,this.unroll=t.unroll==null?!1:t.unroll,this.supportsMasking=!0,this.inputSpec=[new ae({ndim:3})],this.stateSpec=null,this.states_=null,this.numConstants=null,this.keptStates=[]}getStates(){if(this.states_==null){const t=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;return Rn(0,t).map(e=>null)}else return this.states_}setStates(t){this.states_=t}computeOutputShape(t){Ou(t)&&(t=t[0]),t=t;let e=this.cell.stateSize;Array.isArray(e)||(e=[e]);const s=e[0];let o;if(this.returnSequences?o=[t[0],t[1],s]:o=[t[0],s],this.returnState){const r=[];for(const i of e)r.push([t[0],i]);return[o].concat(r)}else return o}computeMask(t,e){return z(()=>{Array.isArray(e)&&(e=e[0]);const s=this.returnSequences?e:null;if(this.returnState){const o=this.states.map(r=>null);return[s].concat(o)}else return s})}get states(){if(this.states_==null){const t=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1,e=[];for(let s=0;s<t;++s)e.push(null);return e}else return this.states_}set states(t){this.states_=t}build(t){if(this.numConstants!=null)throw new yt("Constants support is not implemented in RNN yet.");Ou(t)&&(t=t[0]),t=t;const e=this.stateful?t[0]:null,s=t.slice(2);this.inputSpec[0]=new ae({shape:[e,null,...s]});const o=[t[0]].concat(t.slice(2));this.cell.build(o);let r;if(Array.isArray(this.cell.stateSize)?r=this.cell.stateSize:r=[this.cell.stateSize],this.stateSpec!=null){if(!Tt(this.stateSpec.map(i=>i.shape[i.shape.length-1]),r))throw new F(`An initialState was passed that is not compatible with cell.stateSize. Received stateSpec=${this.stateSpec}; However cell.stateSize is ${this.cell.stateSize}`)}else this.stateSpec=r.map(i=>new ae({shape:[null,i]}));this.stateful&&this.resetStates()}resetStates(t,e=!1){z(()=>{if(!this.stateful)throw new Ln("Cannot call resetStates() on an RNN Layer that is not stateful.");const s=this.inputSpec[0].shape[0];if(s==null)throw new F("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.states_==null)Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(o=>pe([s,o])):this.states_=[pe([s,this.cell.stateSize])];else if(t==null)xt(this.states_),this.keptStates!=null&&(xt(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(o=>pe([s,o])):this.states_[0]=pe([s,this.cell.stateSize]);else{if(Array.isArray(t)||(t=[t]),t.length!==this.states_.length)throw new F(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${t.length} state value(s). Input received: ${t}`);e===!0?this.keptStates.push(this.states_.slice()):xt(this.states_);for(let o=0;o<this.states_.length;++o){const r=t[o],i=Array.isArray(this.cell.stateSize)?this.cell.stateSize[o]:this.cell.stateSize,a=[s,i];if(!Tt(r.shape,a))throw new F(`State ${o} is incompatible with layer ${this.name}: expected shape=${a}, received shape=${r.shape}`);this.states_[o]=r}}this.states_=this.states_.map(o=>Bn(o.clone()))})}apply(t,e){let s=e==null?null:e.initialState,o=e==null?null:e.constants;e==null&&(e={});const r=y0(t,s,o,this.numConstants);t=r.inputs,s=r.initialState,o=r.constants;let i=[],a=[];if(s!=null){e.initialState=s,i=i.concat(s),this.stateSpec=[];for(const c of s)this.stateSpec.push(new ae({shape:c.shape}));a=a.concat(this.stateSpec)}if(o!=null&&(e.constants=o,i=i.concat(o),this.numConstants=o.length),i[0]instanceof qn){const c=[t].concat(i),u=this.inputSpec.concat(a),h=this.inputSpec;this.inputSpec=u;const d=super.apply(c,e);return this.inputSpec=h,d}else return super.apply(t,e)}call(t,e){return z(()=>{const s=e==null?null:e.mask,o=e==null?null:e.training;let r=e==null?null:e.initialState;t=ft(t),r==null&&(this.stateful?r=this.states_:r=this.getInitialState(t));const i=Array.isArray(this.cell.stateSize)?this.cell.stateSize.length:1;if(r.length!==i)throw new F(`RNN Layer has ${i} state(s) but was passed ${r.length} initial state(s).`);this.unroll&&console.warn("Ignoring unroll = true for RNN layer, due to imperative backend.");const a={training:o},c=w0((f,m)=>{const g=this.cell.call([f].concat(m),a);return[g[0],g.slice(1)]},t,r,this.goBackwards,s,null,this.unroll,this.returnSequences),u=c[0],h=c[1],d=c[2];this.stateful&&this.resetStates(d,o);const p=this.returnSequences?h:u;return this.returnState?[p].concat(d):p})}getInitialState(t){return z(()=>{let e=pe(t.shape);return e=it(e,[1,2]),e=ea(e),Array.isArray(this.cell.stateSize)?this.cell.stateSize.map(s=>s>1?Fu(e,[1,s]):e):this.cell.stateSize>1?[Fu(e,[1,this.cell.stateSize])]:[e]})}get trainableWeights(){return this.trainable?this.cell.trainableWeights:[]}get nonTrainableWeights(){return this.trainable?this.cell.nonTrainableWeights:this.cell.weights}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.cell!=null&&this.cell.setFastWeightInitDuringBuild(t)}getConfig(){const t=super.getConfig(),e={returnSequences:this.returnSequences,returnState:this.returnState,goBackwards:this.goBackwards,stateful:this.stateful,unroll:this.unroll};this.numConstants!=null&&(e.numConstants=this.numConstants);const s=this.cell.getConfig();return this.getClassName()===Ws.className&&(e.cell={className:this.cell.getClassName(),config:s}),Object.assign(Object.assign(Object.assign({},s),t),e)}static fromConfig(t,e,s={}){const o=e.cell,r=Tn(o,s);return new t(Object.assign(e,{cell:r}))}}Ws.className="RNN";X(Ws);class zc extends It{}class Ep extends zc{constructor(t){super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=t.units,fe(this.units,"units"),this.activation=Fs(t.activation==null?this.DEFAULT_ACTIVATION:t.activation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Gt(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Gt(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Gt(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=qt(t.kernelRegularizer),this.recurrentRegularizer=qt(t.recurrentRegularizer),this.biasRegularizer=qt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Ko([1,As([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Ko([1,As([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){t=Nt(t),this.kernel=this.addWeight("kernel",[t[t.length-1],this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return z(()=>{if(t=t,t.length!==2)throw new F(`SimpleRNNCell expects 2 input Tensors, got ${t.length}.`);let s=t[1];t=t[0];const o=e.training==null?!1:e.training;0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_s({ones:()=>gn(t),rate:this.dropout,training:o,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_s({ones:()=>gn(s),rate:this.recurrentDropout,training:o,dropoutFunc:this.dropoutFunc}));let r;const i=this.dropoutMask,a=this.recurrentDropoutMask;i!=null?r=Un(A(t,i),this.kernel.read()):r=Un(t,this.kernel.read()),this.bias!=null&&(r=_n(r,this.bias.read())),a!=null&&(s=A(s,a));let l=J(r,Un(s,this.recurrentKernel.read()));return this.activation!=null&&(l=this.activation.apply(l)),[l,l]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:Ds(this.activation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Mt(this.kernelRegularizer),recurrentRegularizer:Mt(this.recurrentRegularizer),biasRegularizer:Mt(this.biasRegularizer),activityRegularizer:Mt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout};return Object.assign(Object.assign({},t),e)}}Ep.className="SimpleRNNCell";X(Ep);class $0 extends Ws{constructor(t){t.cell=new Ep(t),super(t)}call(t,e){return z(()=>{this.cell.dropoutMask!=null&&(xt(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(xt(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return new t(e)}}$0.className="SimpleRNN";X($0);class Rp extends zc{constructor(t){if(super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",t.resetAfter)throw new F("GRUCell does not support reset_after parameter set to true.");this.units=t.units,fe(this.units,"units"),this.activation=Fs(t.activation===void 0?this.DEFAULT_ACTIVATION:t.activation),this.recurrentActivation=Fs(t.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:t.recurrentActivation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Gt(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Gt(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Gt(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelRegularizer=qt(t.kernelRegularizer),this.recurrentRegularizer=qt(t.recurrentRegularizer),this.biasRegularizer=qt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Ko([1,As([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Ko([1,As([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.implementation=t.implementation,this.stateSize=this.units,this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){t=Nt(t);const e=t[t.length-1];this.kernel=this.addWeight("kernel",[e,this.units*3],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units*3],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias?this.bias=this.addWeight("bias",[this.units*3],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint):this.bias=null,this.built=!0}call(t,e){return z(()=>{if(t=t,t.length!==2)throw new F(`GRUCell expects 2 input Tensors (inputs, h, c), got ${t.length}.`);const s=e.training==null?!1:e.training;let o=t[1];t=t[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_s({ones:()=>gn(t),rate:this.dropout,training:s,count:3,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_s({ones:()=>gn(o),rate:this.recurrentDropout,training:s,count:3,dropoutFunc:this.dropoutFunc}));const r=this.dropoutMask,i=this.recurrentDropoutMask;let a,l,c;0<this.dropout&&this.dropout<1&&(t=A(t,r[0]));let u=Un(t,this.kernel.read());this.useBias&&(u=_n(u,this.bias.read())),0<this.recurrentDropout&&this.recurrentDropout<1&&(o=A(o,i[0]));const h=this.recurrentKernel.read(),[d,p]=sn(h,[2*this.units,this.units],h.rank-1),f=Un(o,d),[m,g,x]=sn(u,3,u.rank-1),[b,w]=sn(f,2,f.rank-1);a=this.recurrentActivation.apply(J(m,b)),l=this.recurrentActivation.apply(J(g,w));const y=Un(A(l,o),p);c=this.activation.apply(J(x,y));const $=J(A(a,o),A(J(1,Ht(a)),c));return[$,$]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:Ds(this.activation),recurrentActivation:Ds(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Mt(this.kernelRegularizer),recurrentRegularizer:Mt(this.recurrentRegularizer),biasRegularizer:Mt(this.biasRegularizer),activityRegularizer:Mt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation,resetAfter:!1};return Object.assign(Object.assign({},t),e)}}Rp.className="GRUCell";X(Rp);class C0 extends Ws{constructor(t){t.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),t.cell=new Rp(t),super(t)}call(t,e){return z(()=>{this.cell.dropoutMask!=null&&(xt(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(xt(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return e.implmentation===0&&(e.implementation=1),new t(e)}}C0.className="GRU";X(C0);class Bc extends zc{constructor(t){super(t),this.DEFAULT_ACTIVATION="tanh",this.DEFAULT_RECURRENT_ACTIVATION="hardSigmoid",this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_RECURRENT_INITIALIZER="orthogonal",this.DEFAULT_BIAS_INITIALIZER="zeros",this.units=t.units,fe(this.units,"units"),this.activation=Fs(t.activation===void 0?this.DEFAULT_ACTIVATION:t.activation),this.recurrentActivation=Fs(t.recurrentActivation===void 0?this.DEFAULT_RECURRENT_ACTIVATION:t.recurrentActivation),this.useBias=t.useBias==null?!0:t.useBias,this.kernelInitializer=Gt(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.recurrentInitializer=Gt(t.recurrentInitializer||this.DEFAULT_RECURRENT_INITIALIZER),this.biasInitializer=Gt(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.unitForgetBias=t.unitForgetBias,this.kernelRegularizer=qt(t.kernelRegularizer),this.recurrentRegularizer=qt(t.recurrentRegularizer),this.biasRegularizer=qt(t.biasRegularizer),this.kernelConstraint=ce(t.kernelConstraint),this.recurrentConstraint=ce(t.recurrentConstraint),this.biasConstraint=ce(t.biasConstraint),this.dropout=Ko([1,As([0,t.dropout==null?0:t.dropout])]),this.recurrentDropout=Ko([1,As([0,t.recurrentDropout==null?0:t.recurrentDropout])]),this.dropoutFunc=t.dropoutFunc,this.implementation=t.implementation,this.stateSize=[this.units,this.units],this.dropoutMask=null,this.recurrentDropoutMask=null}build(t){var e;t=Nt(t);const s=t[t.length-1];this.kernel=this.addWeight("kernel",[s,this.units*4],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.recurrentKernel=this.addWeight("recurrent_kernel",[this.units,this.units*4],null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint);let o;if(this.useBias){if(this.unitForgetBias){const r=this.biasInitializer,i=this.units;o=new(e=class extends bn{apply(l,c){const u=r.apply([i]),h=new mp().apply([i]),d=r.apply([i*2]);return Hf(Hf(u,h),d)}},e.className="CustomInit",e)}else o=this.biasInitializer;this.bias=this.addWeight("bias",[this.units*4],null,o,this.biasRegularizer,!0,this.biasConstraint)}else this.bias=null;this.built=!0}call(t,e){return z(()=>{const s=e.training==null?!1:e.training;if(t=t,t.length!==3)throw new F(`LSTMCell expects 3 input Tensors (inputs, h, c), got ${t.length}.`);let o=t[1];const r=t[2];t=t[0],0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_s({ones:()=>gn(t),rate:this.dropout,training:s,count:4,dropoutFunc:this.dropoutFunc})),0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_s({ones:()=>gn(o),rate:this.recurrentDropout,training:s,count:4,dropoutFunc:this.dropoutFunc}));const i=this.dropoutMask,a=this.recurrentDropoutMask;let l,c,u,h;0<this.dropout&&this.dropout<1&&(t=A(t,i[0]));let d=Un(t,this.kernel.read());0<this.recurrentDropout&&this.recurrentDropout<1&&(o=A(o,a[0])),d=J(d,Un(o,this.recurrentKernel.read())),this.useBias&&(d=_n(d,this.bias.read()));const[p,f,m,g]=sn(d,4,d.rank-1);l=this.recurrentActivation.apply(p),c=this.recurrentActivation.apply(f),u=J(A(c,r),A(l,this.activation.apply(m))),h=this.recurrentActivation.apply(g);const x=A(h,this.activation.apply(u));return[x,x,u]})}getConfig(){const t=super.getConfig(),e={units:this.units,activation:Ds(this.activation),recurrentActivation:Ds(this.recurrentActivation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),recurrentInitializer:Kt(this.recurrentInitializer),biasInitializer:Kt(this.biasInitializer),unitForgetBias:this.unitForgetBias,kernelRegularizer:Mt(this.kernelRegularizer),recurrentRegularizer:Mt(this.recurrentRegularizer),biasRegularizer:Mt(this.biasRegularizer),activityRegularizer:Mt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),recurrentConstraint:le(this.recurrentConstraint),biasConstraint:le(this.biasConstraint),dropout:this.dropout,recurrentDropout:this.recurrentDropout,implementation:this.implementation};return Object.assign(Object.assign({},t),e)}}Bc.className="LSTMCell";X(Bc);class I0 extends Ws{constructor(t){t.implementation===0&&console.warn("`implementation=0` has been deprecated, and now defaults to `implementation=1`. Please update your layer call."),t.cell=new Bc(t),super(t)}call(t,e){return z(()=>{this.cell.dropoutMask!=null&&(xt(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(xt(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null);const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}static fromConfig(t,e){return e.implmentation===0&&(e.implementation=1),new t(e)}}I0.className="LSTM";X(I0);class Ap extends zc{constructor(t){super(t),this.cells=t.cells}get stateSize(){const t=[];for(const e of this.cells.slice().reverse())Array.isArray(e.stateSize)?t.push(...e.stateSize):t.push(e.stateSize);return t}call(t,e){return z(()=>{t=t;let s=t.slice(1);const o=[];for(const a of this.cells.slice().reverse())Array.isArray(a.stateSize)?o.push(s.splice(0,a.stateSize.length)):o.push(s.splice(0,1));o.reverse();const r=[];let i;for(let a=0;a<this.cells.length;++a){const l=this.cells[a];s=o[a],a===0?i=[t[0]].concat(s):i=[i[0]].concat(s),i=l.call(i,e),r.push(i.slice(1))}s=[];for(const a of r.slice().reverse())s.push(...a);return[i[0]].concat(s)})}build(t){Ou(t)&&(t=t[0]),t=t;let e;this.cells.forEach((s,o)=>{no(`RNNCell_${o}`,()=>{s.build(t),Array.isArray(s.stateSize)?e=s.stateSize[0]:e=s.stateSize,t=[t[0],e]})}),this.built=!0}getConfig(){const t=super.getConfig(),e=r=>({className:r.getClassName(),config:r.getConfig()}),o={cells:this.cells.map(e)};return Object.assign(Object.assign({},t),o)}static fromConfig(t,e,s={}){const o=[];for(const r of e.cells)o.push(Tn(r,s));return new t({cells:o})}get trainableWeights(){if(!this.trainable)return[];const t=[];for(const e of this.cells)t.push(...e.trainableWeights);return t}get nonTrainableWeights(){const t=[];for(const e of this.cells)t.push(...e.nonTrainableWeights);if(!this.trainable){const e=[];for(const s of this.cells)e.push(...s.trainableWeights);return e.concat(t)}return t}getWeights(){const t=[];for(const e of this.cells)t.push(...e.weights);return Lu(t)}setWeights(t){const e=[];for(const s of this.cells){const o=s.weights.length,r=t.splice(o);for(let i=0;i<s.weights.length;++i)e.push([s.weights[i],r[i]])}Cp(e)}}Ap.className="StackedRNNCells";X(Ap);function _s(n){const{ones:t,rate:e,training:s=!1,count:o=1,dropoutFunc:r}=n,i=()=>r!=null?r(t(),e):fb(t(),e),a=()=>sa(i,t,s);return!o||o<=1?Bn(a().clone()):Array(o).fill(void 0).map(a).map(c=>Bn(c.clone()))}var GA=function(n,t){var e={};for(var s in n)Object.prototype.hasOwnProperty.call(n,s)&&t.indexOf(s)<0&&(e[s]=n[s]);if(n!=null&&typeof Object.getOwnPropertySymbols=="function")for(var o=0,s=Object.getOwnPropertySymbols(n);o<s.length;o++)t.indexOf(s[o])<0&&Object.prototype.propertyIsEnumerable.call(n,s[o])&&(e[s[o]]=n[s[o]]);return e};class v0 extends Ws{constructor(t){if(t.unroll)throw new yt("Unrolling is not possible with convolutional RNNs.");if(Array.isArray(t.cell))throw new yt("It is not possible at the moment to stack convolutional cells.");super(t),this.inputSpec=[new ae({ndim:5})]}call(t,e){return z(()=>{if(this.cell.dropoutMask!=null&&(xt(this.cell.dropoutMask),this.cell.dropoutMask=null),this.cell.recurrentDropoutMask!=null&&(xt(this.cell.recurrentDropoutMask),this.cell.recurrentDropoutMask=null),e&&e.constants)throw new F("ConvRNN2D cell does not support constants");const s=e==null?null:e.mask,o=e==null?null:e.training,r=e==null?null:e.initialState;return super.call(t,{mask:s,training:o,initialState:r})})}computeOutputShape(t){let e=this.computeSingleOutputShape(t);return this.returnSequences||(e=[e[0],...e.slice(2)]),this.returnState&&(e=[e,...Array(2).fill([t[0],...e.slice(-3)])]),e}getInitialState(t){return z(()=>{const{stateSize:e}=this.cell,s=t.shape,o=this.computeSingleOutputShape(s),r=[o[0],...o.slice(2)],i=pe(r);return Array.isArray(e)?Array(e.length).fill(i):[i]})}resetStates(t,e=!1){z(()=>{if(!this.stateful)throw new Ln("Cannot call resetStates() on an RNN Layer that is not stateful.");const s=this.inputSpec[0].shape,o=this.computeSingleOutputShape(s),r=[o[0],...o.slice(2)];if(s[0]==null)throw new F("If an RNN is stateful, it needs to know its batch size. Specify the batch size of your input tensors: \n- If using a Sequential model, specify the batch size by passing a `batchInputShape` option to your first layer.\n- If using the functional API, specify the batch size by passing a `batchShape` option to your Input layer.");if(this.getStates()==null)Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>pe(r)):this.states_=[pe(r)];else if(t==null)xt(this.states_),this.keptStates!=null&&(xt(this.keptStates),this.keptStates=[]),Array.isArray(this.cell.stateSize)?this.states_=this.cell.stateSize.map(()=>pe(r)):this.states_[0]=pe(r);else{if(Array.isArray(t)||(t=[t]),t.length!==this.states_.length)throw new F(`Layer ${this.name} expects ${this.states_.length} state(s), but it received ${t.length} state value(s). Input received: ${t}`);e?this.keptStates.push(this.states_.slice()):xt(this.states_);for(let a=0;a<this.states_.length;++a){const l=t[a],c=r;if(!Tt(l.shape,c))throw new F(`State ${a} is incompatible with layer ${this.name}: expected shape=${c}, received shape=${l.shape}`);this.states_[a]=l}}this.states_=this.states_.map(a=>Bn(a.clone()))})}computeSingleOutputShape(t){const{dataFormat:e,filters:s,kernelSize:o,padding:r,strides:i,dilationRate:a}=this.cell,l=e==="channelsFirst",c=t[l?3:2],u=t[l?4:3],h=En(c,o[0],r,i[0],a[0]),d=En(u,o[1],r,i[1],a[1]);return[...t.slice(0,2),...l?[s,h,d]:[h,d,s]]}}v0.className="ConvRNN2D";class Dp extends Bc{constructor(t){const{filters:e,kernelSize:s,strides:o,padding:r,dataFormat:i,dilationRate:a}=t;super(Object.assign(Object.assign({},t),{units:e})),this.filters=e,fe(this.filters,"filters"),this.kernelSize=Go(s,2,"kernelSize"),this.kernelSize.forEach(l=>fe(l,"kernelSize")),this.strides=Go(o||1,2,"strides"),this.strides.forEach(l=>fe(l,"strides")),this.padding=r||"valid",an(this.padding),this.dataFormat=i||"channelsLast",se(this.dataFormat),this.dilationRate=Go(a||1,2,"dilationRate"),this.dilationRate.forEach(l=>fe(l,"dilationRate"))}build(t){var e;t=Nt(t);const s=this.dataFormat==="channelsFirst"?1:t.length-1;if(t[s]==null)throw new F(`The channel dimension of the input should be defined. Found ${t[s]}`);const o=t[s],r=4,i=this.kernelSize.concat([o,this.filters*r]);this.kernel=this.addWeight("kernel",i,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint);const a=this.kernelSize.concat([this.filters,this.filters*r]);if(this.recurrentKernel=this.addWeight("recurrent_kernel",a,null,this.recurrentInitializer,this.recurrentRegularizer,!0,this.recurrentConstraint),this.useBias){let l;if(this.unitForgetBias){const c=this.biasInitializer,u=this.filters;l=new(e=class extends bn{apply(d,p){const f=c.apply([u]),m=hs([u]),g=c.apply([u*2]);return fp([f,m,g])}},e.className="CustomInit",e)}else l=this.biasInitializer;this.bias=this.addWeight("bias",[this.filters*r],null,l,this.biasRegularizer,!0,this.biasConstraint)}this.built=!0}call(t,e){return z(()=>{if(t.length!==3)throw new F(`ConvLSTM2DCell expects 3 input Tensors (inputs, h, c), got ${t.length}.`);const s=e.training||!1,o=t[0],r=t[1],i=t[2],a=4;0<this.dropout&&this.dropout<1&&this.dropoutMask==null&&(this.dropoutMask=_s({ones:()=>gn(o),rate:this.dropout,training:s,count:a,dropoutFunc:this.dropoutFunc}));const l=this.dropoutMask,c=(j,K,Y)=>!K||!K[Y]?j:A(K[Y],j);let u=c(o,l,0),h=c(o,l,1),d=c(o,l,2),p=c(o,l,3);0<this.recurrentDropout&&this.recurrentDropout<1&&this.recurrentDropoutMask==null&&(this.recurrentDropoutMask=_s({ones:()=>gn(r),rate:this.recurrentDropout,training:s,count:a,dropoutFunc:this.dropoutFunc}));const f=this.recurrentDropoutMask;let m=c(r,f,0),g=c(r,f,1),x=c(r,f,2),b=c(r,f,3);const w=3,[y,$,v,N]=sn(this.kernel.read(),a,w),[T,k,S,C]=this.useBias?sn(this.bias.read(),a):[null,null,null,null];u=this.inputConv(u,y,T,this.padding),h=this.inputConv(h,$,k,this.padding),d=this.inputConv(d,v,S,this.padding),p=this.inputConv(p,N,C,this.padding);const[D,O,P,M]=sn(this.recurrentKernel.read(),a,w);m=this.recurrentConv(m,D),g=this.recurrentConv(g,O),x=this.recurrentConv(x,P),b=this.recurrentConv(b,M);const V=this.recurrentActivation.apply(J(u,m)),H=this.recurrentActivation.apply(J(h,g)),W=J(A(H,i),A(V,this.activation.apply(J(d,x)))),q=A(this.recurrentActivation.apply(J(p,b)),this.activation.apply(W));return[q,q,W]})}getConfig(){const t=super.getConfig(),{units:e}=t,s=GA(t,["units"]),o={filters:this.filters,kernelSize:this.kernelSize,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,strides:this.strides};return Object.assign(Object.assign({},s),o)}inputConv(t,e,s,o){const r=ho(t,e,this.strides,o||"valid",this.dataFormat==="channelsFirst"?"NCHW":"NHWC",this.dilationRate);return s?_n(r,s,this.dataFormat):r}recurrentConv(t,e){return ho(t,e,1,"same",this.dataFormat==="channelsFirst"?"NCHW":"NHWC")}}Dp.className="ConvLSTM2DCell";X(Dp);class k0 extends v0{constructor(t){const e=new Dp(t);super(Object.assign(Object.assign({},t),{cell:e}))}static fromConfig(t,e){return new t(e)}}k0.className="ConvLSTM2D";X(k0);class Fp extends It{constructor(t){super(t),this.rate=Math.max(Math.min(t.rate,1),0),this.noiseShape=t.noiseShape,this.seed=t.seed,this.supportsMasking=!0}getNoiseShape(t){if(this.noiseShape==null)return this.noiseShape;const e=t.shape,s=[];for(let o=0;o<this.noiseShape.length;++o)s.push(this.noiseShape[o]==null?e[o]:this.noiseShape[o]);return s}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t);if(0<this.rate&&this.rate<1){const o=e.training==null?!1:e.training,r=this.getNoiseShape(s);return sa(()=>fb(s,this.rate,r,this.seed),()=>s,o)}return t})}getConfig(){const t={rate:this.rate,noiseShape:this.noiseShape,seed:this.seed},e=super.getConfig();return Object.assign(t,e),t}dispose(){return super.dispose()}}Fp.className="Dropout";X(Fp);class S0 extends Fp{constructor(t){super(t),this.inputSpec=[{ndim:3}]}getNoiseShape(t){const e=t.shape;return[e[0],1,e[2]]}}S0.className="SpatialDropout1D";X(S0);class N0 extends It{constructor(t){if(super(t),this.activation=null,this.useBias=!0,this.kernel=null,this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",t.batchInputShape==null&&t.inputShape==null&&t.inputDim!=null){let e=null;t.batchSize!=null&&(e=t.batchSize),this.batchInputShape=[e,t.inputDim]}this.units=t.units,fe(this.units,"units"),this.activation=Fs(t.activation),t.useBias!=null&&(this.useBias=t.useBias),this.kernelInitializer=Gt(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.biasInitializer=Gt(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.kernelConstraint=ce(t.kernelConstraint),this.biasConstraint=ce(t.biasConstraint),this.kernelRegularizer=qt(t.kernelRegularizer),this.biasRegularizer=qt(t.biasRegularizer),this.activityRegularizer=qt(t.activityRegularizer),this.supportsMasking=!0,this.inputSpec=[{minNDim:2}]}build(t){t=Nt(t);const e=t[t.length-1];this.kernel==null&&(this.kernel=this.addWeight("kernel",[e,this.units],null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.units],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint))),this.inputSpec=[{minNDim:2,axes:{[-1]:e}}],this.built=!0}computeOutputShape(t){t=Nt(t);const e=t.slice();return e[e.length-1]=this.units,e}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t),o=ab(this.activation.getClassName());let r;return o!=null?r=Un(s,this.kernel.read(),o,this.bias?this.bias.read():null):(r=Un(s,this.kernel.read()),this.bias!=null&&(r=_n(r,this.bias.read())),this.activation!=null&&(r=this.activation.apply(r))),r})}getConfig(){const t={units:this.units,activation:Ds(this.activation),useBias:this.useBias,kernelInitializer:Kt(this.kernelInitializer),biasInitializer:Kt(this.biasInitializer),kernelRegularizer:Mt(this.kernelRegularizer),biasRegularizer:Mt(this.biasRegularizer),activityRegularizer:Mt(this.activityRegularizer),kernelConstraint:le(this.kernelConstraint),biasConstraint:le(this.biasConstraint)},e=super.getConfig();return Object.assign(t,e),t}}N0.className="Dense";X(N0);class T0 extends It{constructor(t){t=t||{},super(t),this.inputSpec=[{minNDim:3}],this.dataFormat=t.dataFormat}computeOutputShape(t){t=Nt(t);for(const e of t.slice(1))if(e==null)throw new F(`The shape of the input to "Flatten" is not fully defined (got ${t.slice(1)}). Make sure to pass a complete "input_shape" or "batch_input_shape" argument to the first layer in your model.`);return[t[0],Ss(t,1)]}call(t,e){return z(()=>{this.invokeCallHook(t,e);let s=ft(t);if(this.dataFormat==="channelsFirst"&&s.rank>1){const o=[0];for(let r=2;r<s.rank;++r)o.push(r);o.push(1),s=vt(s,o)}return IR(s)})}getConfig(){const t={};this.dataFormat!=null&&(t.dataFormat=this.dataFormat);const e=super.getConfig();return Object.assign(t,e),t}}T0.className="Flatten";X(T0);class E0 extends It{constructor(t){super(t),this.supportsMasking=!0,this.activation=Fs(t.activation)}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t);return this.activation.apply(s)})}getConfig(){const t={activation:Ds(this.activation)},e=super.getConfig();return Object.assign(t,e),t}}E0.className="Activation";X(E0);class R0 extends It{constructor(t){super(t),this.n=t.n,this.inputSpec=[{ndim:2}]}computeOutputShape(t){return[t[0],this.n,t[1]]}call(t,e){return z(()=>(t=ft(t),$R(t,this.n)))}getConfig(){const t={n:this.n},e=super.getConfig();return Object.assign(t,e),t}}R0.className="RepeatVector";X(R0);class A0 extends It{constructor(t){super(t),this.targetShape=t.targetShape;for(let e=0;e<this.targetShape.length;++e)this.isUnknown(this.targetShape[e])&&(this.targetShape[e]=null)}isUnknown(t){return t<0||t==null}fixUnknownDimension(t,e){const s="Total size of new array must be unchanged.",o=e.slice();let r=1,i=null;for(let l=0;l<o.length;++l){const c=o[l];if(this.isUnknown(c))if(i===null)i=l;else throw new F("Can only specifiy one unknown dimension.");else r*=c}const a=Ss(t);if(i!==null){if(r===0||a%r!==0)throw new F(s);o[i]=a/r}else if(a!==r)throw new F(s);return o}computeOutputShape(t){let e=!1;for(let s=0;s<t.length;++s)if(this.isUnknown(t[s])){e=!0;break}return e?t.slice(0,1).concat(this.targetShape):t.slice(0,1).concat(this.fixUnknownDimension(t.slice(1),this.targetShape))}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t),o=s.shape,r=o.slice(0,1).concat(this.fixUnknownDimension(o.slice(1),this.targetShape));return L(s,r)})}getConfig(){const t={targetShape:this.targetShape},e=super.getConfig();return Object.assign(t,e),t}}A0.className="Reshape";X(A0);class D0 extends It{constructor(t){if(super(t),t.dims==null)throw new Error("Required configuration field `dims` is missing during Permute constructor call.");if(!Array.isArray(t.dims))throw new Error(`Permute constructor requires \`dims\` to be an Array, but received ${t.dims} instead.`);const e=Rn(1,t.dims.length+1);if(!Tt(t.dims.slice().sort(),e))throw new Error("Invalid permutation `dims`: "+JSON.stringify(t.dims)+" `dims` must contain consecutive integers starting from 1.");this.dims=t.dims,this.dimsIncludingBatch=[0].concat(this.dims),this.inputSpec=[new ae({ndim:this.dims.length+1})]}computeOutputShape(t){t=Nt(t);const e=t.slice();return this.dims.forEach((s,o)=>{e[o+1]=t[s]}),e}call(t,e){return vt(ft(t),this.dimsIncludingBatch)}getConfig(){const t={dims:this.dims},e=super.getConfig();return Object.assign(t,e),t}}D0.className="Permute";X(D0);class F0 extends It{constructor(t){super(t??{}),this.supportsMasking=!0,t!=null?this.maskValue=t.maskValue==null?0:t.maskValue:this.maskValue=0}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={maskValue:this.maskValue};return Object.assign(e,t),e}computeMask(t,e){const s=ft(t);return Su(Lr(s,this.maskValue),-1)}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t),i=Su(Lr(s,this.maskValue),-1,!0);return A(s,st(i,s.dtype))})}}F0.className="Masking";X(F0);class _0 extends It{constructor(t){if(super(t),this.embeddings=null,this.DEFAULT_EMBEDDINGS_INITIALIZER="randomUniform",t.batchInputShape==null&&t.inputShape==null){let e=null;t.batchSize!=null&&(e=t.batchSize),t.inputLength==null?this.batchInputShape=[e,null]:this.batchInputShape=[e].concat(At(t.inputLength))}this.inputDim=t.inputDim,fe(this.inputDim,"inputDim"),this.outputDim=t.outputDim,fe(this.outputDim,"outputDim"),this.embeddingsInitializer=Gt(t.embeddingsInitializer||this.DEFAULT_EMBEDDINGS_INITIALIZER),this.embeddingsRegularizer=qt(t.embeddingsRegularizer),this.activityRegularizer=qt(t.activityRegularizer),this.embeddingsConstraint=ce(t.embeddingsConstraint),this.maskZero=t.maskZero,this.supportsMasking=t.maskZero,this.inputLength=t.inputLength}build(t){this.embeddings=this.addWeight("embeddings",[this.inputDim,this.outputDim],this.dtype,this.embeddingsInitializer,this.embeddingsRegularizer,!0,this.embeddingsConstraint),this.built=!0}warnOnIncompatibleInputShape(t){}computeMask(t,e){return z(()=>this.maskZero?(t=ft(t),Lr(t,kt(t))):null)}computeOutputShape(t){if(t=Nt(t),this.inputLength==null)return[...t,this.outputDim];const e=At(this.inputLength);if(e.length!==t.length-1)throw new F(`"inputLength" is ${this.inputLength}, but received input shape has shape ${t}`);{let s=0;for(let o=0;o<e.length;++o){const r=e[o],i=t[o+1];if(r!=null&&i!=null&&r!==i)throw new F(`"inputLength" is ${this.inputLength}, but received input shape has shape ${t}`);r==null&&(e[s]=i),s++}}return[t[0],...e,this.outputDim]}call(t,e){return z(()=>{this.invokeCallHook(t,e);let s=ft(t);s.dtype!=="int32"&&(s=Wn(s,"int32"));const o=pb(this.embeddings.read(),L(s,[s.size]));return L(o,Nt(this.computeOutputShape(s.shape)))})}getConfig(){const t={inputDim:this.inputDim,outputDim:this.outputDim,embeddingsInitializer:Kt(this.embeddingsInitializer),embeddingsRegularizer:Mt(this.embeddingsRegularizer),activityRegularizer:Mt(this.activityRegularizer),embeddingsConstraint:le(this.embeddingsConstraint),maskZero:this.maskZero,inputLength:this.inputLength},e=super.getConfig();return Object.assign(t,e),t}}_0.className="Embedding";X(_0);class Eo extends It{constructor(t){super(t||{}),this.supportsMasking=!0}mergeFunction(t){throw new yt}computeElementwiseOpOutputShape(t,e){if(t==null||e==null)return null;if(t.length<e.length)return this.computeElementwiseOpOutputShape(e,t);if(e.length===0)return t;const s=t.slice(0,t.length-e.length);for(let o=0;o<e.length;++o){const r=t[t.length-e.length+o],i=e[o];if(r==null||i==null||r<0||i<0)s.push(null);else if(r===1)s.push(i);else if(i===1)s.push(r);else{if(r!==i)throw new F("Operands could not be broadcast together with shapes "+JSON.stringify(t)+" "+JSON.stringify(e));s.push(r)}}return s}build(t){if(Array.isArray(t)&&!Array.isArray(t[0])&&(t=[Nt(t)]),t=t,t.length<2)throw new F(`A merge layer should be called on an Array of at least 2 inputs. Got ${t.length} input(s).`);let e=[];for(const r of t)r!=null&&r[0]!==null&&e.push(r[0]);if(e=ks(e),e.length>1)throw new F(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(t)}.`);let s=t[0]==null?null:t[0].slice(1);for(let r=1;r<t.length;++r){const i=t[r]==null?null:t[r].slice(1);s=this.computeElementwiseOpOutputShape(s,i)}const o=t.map(r=>r.length);t.indexOf(null)===-1&&ks(o).length===1?this.reshapeRequired=!1:this.reshapeRequired=!0}call(t,e){return z(()=>{if(t=t,this.reshapeRequired){const s=[],o=t.map(r=>r.rank);if(o.indexOf(null)===-1){const r=As(o);for(let i of t){const a=i.rank;for(let l=0;l<r-a;++l)i=ea(i,1);s.push(i)}return this.mergeFunction(s)}else{let r=!1;for(const l of t){const c=l.rank;if(c==null){const u=l.shape,h=u[0],d=u.slice(1).concat([h]);let p=L(l,[h].concat(Ss(u.slice(1))));p=vt(p,[1,0]),p=L(p,d),s.push(p),r=!0}else if(c>1){const u=Rn(1,c).concat([0]);s.push(vt(l,u)),r=!0}else s.push(l)}let i=this.mergeFunction(s);const a=i.rank;if(r){if(a==null){const l=i.shape,c=l.length,u=l[c-1],h=[u].concat(l.slice(0,l.length-1));i=L(vt(L(i,[-1,u]),[1,0]),h)}else if(a>1){const l=[a-1].concat(Rn(0,a-1));i=vt(i,l)}}return i}}else return this.mergeFunction(t)})}computeOutputShape(t){t=t;let e;t[0]==null?e=null:e=t[0].slice(1);for(let o=1;o<t.length;++o){const r=t[o]==null?null:t[o].slice(1);e=this.computeElementwiseOpOutputShape(e,r)}let s=[];for(const o of t)o!=null&&o[0]!==null&&s.push(o[0]);return s=ks(s),s.length===1?e=s.concat(e):e=[null].concat(e),e}computeMask(t,e){return z(()=>{if(e==null)return null;if(!Array.isArray(e))throw new F("`mask` should be an Array");if(!Array.isArray(t))throw new F("`inputs` should be an Array");if(e.length!==t.length)throw new F(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${t.length} vs ${e.length})`);if(e.every(o=>o==null))return null;e=e.map(o=>o==null?o:Me(o,0));let s=e[0];for(let o=1;o<e.length-1;++o)s=Gn(s,e[o]);return s})}}class O0 extends Eo{constructor(t){super(t)}mergeFunction(t){return z(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=J(e,t[s]);return e})}}O0.className="Add";X(O0);class L0 extends Eo{constructor(t){super(t)}mergeFunction(t){return z(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=A(e,t[s]);return e})}}L0.className="Multiply";X(L0);class M0 extends Eo{constructor(t){super(t)}mergeFunction(t){return z(()=>{let e=t[0].clone();for(let s=1;s<t.length;++s)e=J(e,t[s]);return A(1/t.length,e)})}}M0.className="Average";X(M0);class P0 extends Eo{constructor(t){super(t)}mergeFunction(t){return z(()=>{let e=t[0];for(let s=1;s<t.length;++s)e=zs(e,t[s]);return e})}}P0.className="Maximum";X(P0);class z0 extends Eo{constructor(t){super(t)}mergeFunction(t){return z(()=>{let e=t[0];for(let s=1;s<t.length;++s)e=jo(e,t[s]);return e})}}z0.className="Minimum";X(z0);class B0 extends Eo{constructor(t){super(t),this.DEFAULT_AXIS=-1,t==null&&(t={}),this.axis=t.axis==null?this.DEFAULT_AXIS:t.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(t){if(!(Array.isArray(t)&&Array.isArray(t[0]))||t.length===1)throw new F("A `Concatenate` layer should be called on a list of at least 2 inputs");t=t;let e=!0;for(const o of t)if(o!=null){e=!1;break}if(e)return;const s=[];for(let o=0;o<t.length;++o){const r=t[o].slice();r.splice(this.axis,1);let i=!1;for(const a of s)if(Tt(a,r)){i=!0;break}i||s.push(r)}if(s.length>1)throw new F("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(t))}mergeFunction(t){return z(()=>fp(t,this.axis))}computeOutputShape(t){if(!(Array.isArray(t)&&Array.isArray(t[0])))throw new F("A `Concatenate` layer should be called on a list of inputs.");const e=t,s=e[0].slice(),o=this.axis<0?s.length+this.axis:this.axis;for(const r of e.slice(1)){if(s[o]==null||r[o]==null){s[o]=null;break}s[o]+=r[o]}return s}computeMask(t,e){if(e==null)return null;if(!Array.isArray(e))throw new F("`mask` should be an array for Concatenate");if(!Array.isArray(t))throw new F("`inputs` should be an array for Concatenate");if(e.length!==t.length)throw new F(`Mismatch in the length of mask (${e.length}) and the legnth of inputs (${t.length})`);return z(()=>{let s=!0;if(e.forEach(i=>{if(i!=null){s=!1;return}}),s)return null;const o=[];for(let i=0;i<t.length;++i)e[i]==null?o.push(st(gn(t[i]),"bool")):e[i].rank<t[i].rank?o.push(Me(e[i],-1)):o.push(e[i]);const r=ke(o,this.axis);return Mg(r,-1,!1)})}getConfig(){const t={axis:this.axis},e=super.getConfig();return Object.assign(t,e),t}}B0.className="Concatenate";X(B0);function yr(n,t){for(;n<0;)n+=t;return n}function HA(n,t,e){if(n.shape.length>3||t.shape.length>3)throw new yt("batchDot is not implemented for tensors of 4D or higher rank yet");if(I(n.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${n.shape.length}`),I(n.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${t.shape.length}`),typeof e=="number"&&(e=[e,e]),n.dtype==="complex64"||t.dtype==="complex64")throw new yt("batchDot is not implemented for complex64-type Tensors yet.");const s=n.shape.length,o=t.shape.length;e==null&&(e=[s-1,o-2]);const r=e;return z(()=>{let i;if(s>o){i=s-o;const l=[];for(let c=0;c<i;++c)l.push(1);t=L(t,t.shape.concat(l))}else if(o>s){i=o-s;const l=[];for(let c=0;c<i;++c)l.push(1);n=L(n,n.shape.concat(l))}else i=0;let a;if(n.shape.length===2&&t.shape.length===2)r[0]===r[1]?a=it(A(n,t),r[0]):a=it(A(vt(n,[1,0]),t),r[1]);else{const l=r[0]!==n.shape.length-1,c=r[1]===t.shape.length-1;a=Et(n,t,l,c)}if(i>0){let l;s>o?l=s+o-3:l=s-1;const c=[];for(let u=l;u<l+i;++u)c.push(u);a=Xi(a,c)}return a.shape.length===1&&(a=Me(a,1)),a})}class V0 extends Eo{constructor(t){super(t),this.axes=t.axes,this.normalize=t.normalize==null?!1:t.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(t){I(Array.isArray(t)&&t.length===2&&Array.isArray(t[0])&&Array.isArray(t[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const e=t[0],s=t[1];if(e.length>3||s.length>3)throw new yt("Dot layer does not support tensors of 4D or higher rank yet.");const o=this.interpretAxes(e,s);if(e[o[0]]!==s[o[1]])throw new F(`Dimension incompatibility: ${e[o[0]]} !== ${s[o[1]]}`)}mergeFunction(t){if(t.length!==2)throw new F(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${t.length} input(s).`);let e=t[0],s=t[1],o;return Array.isArray(this.axes)?o=this.axes.map((r,i)=>yr(r,t[i].shape.length)):o=[yr(this.axes,e.shape.length),yr(this.axes,s.shape.length)],this.normalize&&(e=tl(e,o[0]),s=tl(s,o[1])),HA(e,s,o)}interpretAxes(t,e){let s;return Array.isArray(this.axes)?s=this.axes:s=[yr(this.axes,t.length),yr(this.axes,e.length)],s}computeOutputShape(t){I(Array.isArray(t)&&t.length===2&&Array.isArray(t[0])&&Array.isArray(t[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const e=t[0].slice(),s=t[1].slice();if(e.length>3||s.length>3)throw new yt("Dot layer does not support tensors of 4D or higher rank yet.");const o=this.interpretAxes(e,s);e.splice(o[0],1),s.splice(o[1],1),s.splice(0,1);const r=e.concat(s);return r.length===1&&r.push(1),r}computeMask(t,e){return null}getConfig(){const t={axes:this.axes,normalize:this.normalize},e=super.getConfig();return Object.assign(t,e),t}}V0.className="Dot";X(V0);class W0 extends It{constructor(t){super(t),this.supportsMasking=!0,this.stddev=t.stddev}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={stddev:this.stddev};return Object.assign(e,t),e}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t);return sa(()=>J(Dc(s.shape,0,this.stddev),s),()=>s,e.training||!1)})}}W0.className="GaussianNoise";X(W0);class U0 extends It{constructor(t){super(t),this.supportsMasking=!0,this.rate=t.rate}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={rate:this.rate};return Object.assign(e,t),e}call(t,e){return z(()=>{this.invokeCallHook(t,e);const s=ft(t);return this.rate>0&&this.rate<1?sa(()=>{const r=Math.sqrt(this.rate/(1-this.rate));return A(s,Dc(s.shape,1,r))},()=>s,e.training||!1):s})}}U0.className="GaussianDropout";X(U0);class G0 extends It{constructor(t){super(t),this.supportsMasking=!0,this.rate=t.rate,this.noiseShape=t.noiseShape}_getNoiseShape(t){return this.noiseShape||ft(t).shape}computeOutputShape(t){return t}getConfig(){const t=super.getConfig(),e={rate:this.rate};return Object.assign(e,t),e}call(t,e){return z(()=>{if(this.rate<1&&this.rate>0){const s=this._getNoiseShape(t);return sa(()=>{const r=ft(t),a=-1.6732632423543772*1.0507009873554805;let l=So(Ki(s),this.rate);l=Wn(l,"float32");const c=((1-this.rate)*(1+this.rate*a**2))**-.5,u=-c*a*this.rate,h=J(A(r,l),A(J(l,-1),a));return J(A(h,c),u)},()=>ft(t),e.training||!1)}return t})}}G0.className="AlphaDropout";X(G0);function Wr(n,t,e,s,o,r=.001){let i;if(n.rank===2)i=EI(n,t,e,s,o,r);else if(n.rank===3)i=AI(n,t,e,s,o,r);else if(n.rank===4)i=FI(n,t,e,s,o,r);else throw new yt(`batchNormalization is not implemented for array of rank ${n.rank} yet`);return i}function qA(n,t,e,s,o=.001){return z(()=>{const r=yd(n,s),i=r.mean,a=r.variance;return[Wr(n,i,a,e,t,o),i,a]})}function jA(n,t,e,s,o=.001){return z(()=>{const r=yd(n,s),i=r.mean,a=r.variance,l=[];for(const f of Rn(0,n.rank))s.indexOf(f)!==-1?l.push(1):l.push(n.shape[f]);const c=L(i,l),u=L(a,l),h=t==null?null:L(t,l),d=e==null?null:L(e,l);return[Wr(n,c,u,d,h,o),i,a]})}function KA(n,t,e,s,o=.001){return Tt(s.slice().sort(),Rn(0,n.rank-1))?qA(n,t,e,s,o):jA(n,t,e,s,o)}class H0 extends It{constructor(t){t==null&&(t={}),super(t),this.supportsMasking=!0,this.axis=t.axis==null?-1:t.axis,this.momentum=t.momentum==null?.99:t.momentum,this.epsilon=t.epsilon==null?.001:t.epsilon,this.center=t.center==null?!0:t.center,this.scale=t.scale==null?!0:t.scale,this.betaInitializer=Gt(t.betaInitializer||"zeros"),this.gammaInitializer=Gt(t.gammaInitializer||"ones"),this.movingMeanInitializer=Gt(t.movingMeanInitializer||"zeros"),this.movingVarianceInitializer=Gt(t.movingVarianceInitializer||"ones"),this.betaConstraint=ce(t.betaConstraint),this.gammaConstraint=ce(t.gammaConstraint),this.betaRegularizer=qt(t.betaRegularizer),this.gammaRegularizer=qt(t.gammaRegularizer)}build(t){t=Nt(t);const e=this.axis>=0?this.axis:this.axis+t.length,s=t[e];if(s==null)throw new F(`Axis ${e} of input tensor should have a defined dimension but the layer received an input with shape ${JSON.stringify(t)}.`);this.inputSpec=[new ae({ndim:t.length,axes:{[e]:s}})];const o=[s];this.scale&&(this.gamma=this.addWeight("gamma",o,null,this.gammaInitializer,this.gammaRegularizer,!0,this.gammaConstraint)),this.center&&(this.beta=this.addWeight("beta",o,null,this.betaInitializer,this.betaRegularizer,!0,this.betaConstraint)),this.movingMean=this.addWeight("moving_mean",o,null,this.movingMeanInitializer,null,!1),this.movingVariance=this.addWeight("moving_variance",o,null,this.movingVarianceInitializer,null,!1),this.built=!0}call(t,e){return z(()=>{const s=e.training==null?!1:e.training,o=ft(t),r=o.shape,i=r.length,a=Rn(0,i),l=this.axis>=0?this.axis:this.axis+i;a.splice(l,1);const c=bo(1,i);c[l]=r[l];const u=a.slice();u.sort();const h=!Tt(u,Rn(0,i).slice(0,i-1)),d=()=>{if(h){const b=L(this.movingMean.read(),c),w=L(this.movingVariance.read(),c),y=this.center?L(this.beta.read(),c):null,$=this.scale?L(this.gamma.read(),c):null;return Wr(o,b,w,y,$,this.epsilon)}else return Wr(o,this.movingMean.read(),this.movingVariance.read(),this.beta==null?null:this.beta.read(),this.gamma==null?null:this.gamma.read(),this.epsilon)};if(!s)return d();const[p,f,m]=KA(o,this.gamma.read(),this.beta.read(),a,this.epsilon),g=(b,w,y)=>{z(()=>{const $=1-y,v=b.read(),N=A(rt(v,w),$);b.write(rt(v,N))})};return g(this.movingMean,f,this.momentum),g(this.movingVariance,m,this.momentum),p})}getConfig(){const t={axis:this.axis,momentum:this.momentum,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Kt(this.betaInitializer),gammaInitializer:Kt(this.gammaInitializer),movingMeanInitializer:Kt(this.movingMeanInitializer),movingVarianceInitializer:Kt(this.movingVarianceInitializer),betaRegularizer:Mt(this.betaRegularizer),gammaRegularizer:Mt(this.gammaRegularizer),betaConstraint:le(this.betaConstraint),gammaConstraint:le(this.gammaConstraint)},e=super.getConfig();return Object.assign(t,e),t}}H0.className="BatchNormalization";X(H0);class q0 extends It{constructor(t){if(t==null&&(t={}),super(t),this.axis=t.axis==null?-1:t.axis,typeof this.axis=="number"){if(!Number.isInteger(this.axis))throw new Error(`Expected axis to be an integer, but received ${this.axis}`)}else if(Array.isArray(this.axis)){for(const e of this.axis)if(!Number.isInteger(e))throw new Error(`Expected axis to be an array of integers, but received ${JSON.stringify(this.axis)}`)}else throw new Error(`Expected axis to be an integer or an array of integers, but received ${JSON.stringify(this.axis)}`);this.epsilon=t.epsilon==null?.001:t.epsilon,this.center=t.center==null?!0:t.center,this.scale=t.scale==null?!0:t.scale,this.betaInitializer=Gt(t.betaInitializer||"zeros"),this.gammaInitializer=Gt(t.gammaInitializer||"ones"),this.betaRegularizer=qt(t.betaRegularizer),this.gammaRegularizer=qt(t.gammaRegularizer),this.supportsMasking=!0}build(t){t=Nt(t);const e=t.length;typeof this.axis=="number"&&(this.axis=[this.axis]);for(let r=0;r<this.axis.length;++r)this.axis[r]<0&&(this.axis[r]+=e);for(const r of this.axis)if(r<0||r>=e)throw new Error(`Invalid axis: ${r}`);if(this.axis.length!==ks(this.axis).length)throw new Error(`Found duplicate axes in: ${this.axis}`);const s=this.axis.map(r=>t[r]),o=!0;this.scale?this.gamma=this.addWeight("gamma",s,"float32",this.gammaInitializer,this.gammaRegularizer,o):this.gamma=null,this.center?this.beta=this.addWeight("beta",s,"float32",this.betaInitializer,this.betaRegularizer,o):this.beta=null,this.built=!0}call(t,e){const s=ft(t),o=s.shape,r=o.length;return z(()=>{let{mean:a,variance:l}=yd(s,this.axis,!0);const c=bo(1,r);for(const m of this.axis)c[m]=o[m];const u=m=>m!=null&&m.shape.length!==r?L(m,c):m;let h=this.scale?u(this.gamma.read()):null,d=this.center?u(this.beta.read()):null;const p=[],f=[];for(let m=0;m<r;++m)this.axis.indexOf(m)!==-1?(p.push(o[m]),f.push(1)):(p.push(1),f.push(o[m]));return a=kn(a,p),l=kn(l,p),h!=null&&(h=kn(h,f)),d!=null&&(d=kn(d,f)),Wr(s,a,l,d,h,this.epsilon)})}getConfig(){const t={axis:this.axis,epsilon:this.epsilon,center:this.center,scale:this.scale,betaInitializer:Kt(this.betaInitializer),gammaInitializer:Kt(this.gammaInitializer),betaRegularizer:Mt(this.betaRegularizer),gammaRegularizer:Mt(this.gammaRegularizer)},e=super.getConfig();return Object.assign(t,e),t}}q0.className="LayerNormalization";X(q0);function XA(n,t,e){return z(()=>{if(n.rank!==4)throw new F(`temporalPadding expects input tensor to be 4-D, but received a ${n.rank}-D tensor.`);if(t==null&&(t=[[1,1],[1,1]]),t.length!==2||t[0].length!==2||t[1].length!==2)throw new F("spatial2dPadding expects `padding` to be an Array of two Arrays, each of which is an Array of two integers.");if(e==null&&(e=An()),e!=="channelsLast"&&e!=="channelsFirst")throw new F(`Unknown data format: ${e}. Supported data formats are 'channelsLast' and 'channelsFirst.`);let s;return e==="channelsFirst"?s=[[0,0],[0,0],t[0],t[1]]:s=[[0,0],t[0],t[1],[0,0]],wd(n,s)})}class j0 extends It{constructor(t){if(t==null&&(t={}),super(t),this.dataFormat=t.dataFormat==null?An():t.dataFormat,t.padding==null)this.padding=[[1,1],[1,1]];else if(typeof t.padding=="number")this.padding=[[t.padding,t.padding],[t.padding,t.padding]];else{if(t.padding=t.padding,t.padding.length!==2)throw new F(`ZeroPadding2D expects padding to be a length-2 array, but received a length-${t.padding.length} array.`);let e,s;if(typeof t.padding[0]=="number")e=[t.padding[0],t.padding[0]],s=[t.padding[1],t.padding[1]];else{if(t.padding=t.padding,t.padding[0].length!==2)throw new F(`ZeroPadding2D expects height padding to be a length-2 array, but received a length-${t.padding[0].length} array.`);if(e=t.padding[0],t.padding[1].length!==2)throw new F(`ZeroPadding2D expects width padding to be a length-2 array, but received a length-${t.padding[1].length} array.`);s=t.padding[1]}this.padding=[e,s]}this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){t=Nt(t);let e,s;return this.dataFormat==="channelsFirst"?(t[2]!=null&&t[2]>=0?e=t[2]+this.padding[0][0]+this.padding[0][1]:e=null,t[3]!=null&&t[3]>=0?s=t[3]+this.padding[1][0]+this.padding[1][1]:s=null,[t[0],t[1],e,s]):(t[1]!=null&&t[1]>=0?e=t[1]+this.padding[0][0]+this.padding[0][1]:e=null,t[2]!=null&&t[2]>=0?s=t[2]+this.padding[1][0]+this.padding[1][1]:s=null,[t[0],e,s,t[3]])}call(t,e){return z(()=>XA(ft(t),this.padding,this.dataFormat))}getConfig(){const t={padding:this.padding,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}j0.className="ZeroPadding2D";X(j0);function Vc(n,t,e,s,o,r){return z(()=>{se(o),cb(r),an(s),e==null&&(e=[1,1]),s==null&&(s="valid"),o==null&&(o=An()),r==null&&(r="max"),n=Tp(n,o);let i;const a=s==="same"?"same":"valid";return r==="max"?i=bd(n,t,e,a):i=id(n,t,e,a),o==="channelsFirst"&&(i=vt(i,[0,3,1,2])),i})}function K0(n,t,e,s,o,r){return z(()=>{se(o),cb(r),an(s),e==null&&(e=[1,1,1]),s==null&&(s="valid"),o==null&&(o=An()),r==null&&(r="max"),n=h0(n,o);let i;const a=s==="same"?"same":"valid";return r==="max"?i=xk(n,t,e,a):i=yI(n,t,e,a),o==="channelsFirst"&&(i=vt(i,[0,4,1,2,3])),i})}class X0 extends It{constructor(t){if(t.poolSize==null&&(t.poolSize=2),super(t),typeof t.poolSize=="number")this.poolSize=[t.poolSize];else if(Array.isArray(t.poolSize)&&t.poolSize.length===1&&typeof t.poolSize[0]=="number")this.poolSize=t.poolSize;else throw new F(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(t.poolSize)}`);if(fe(this.poolSize,"poolSize"),t.strides==null)this.strides=this.poolSize;else if(typeof t.strides=="number")this.strides=[t.strides];else if(Array.isArray(t.strides)&&t.strides.length===1&&typeof t.strides[0]=="number")this.strides=t.strides;else throw new F(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(t.strides)}`);fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,an(this.padding),this.inputSpec=[new ae({ndim:3})]}computeOutputShape(t){t=Nt(t);const e=En(t[1],this.poolSize[0],this.padding,this.strides[0]);return[t[0],e,t[2]]}call(t,e){return z(()=>{this.invokeCallHook(t,e),t=ea(ft(t),2);const s=this.poolingFunction(ft(t),[this.poolSize[0],1],[this.strides[0],1],this.padding,"channelsLast");return Xi(s,[2])})}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides},e=super.getConfig();return Object.assign(t,e),t}}class Y0 extends X0{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),Vc(t,e,s,o,r,"max")}}Y0.className="MaxPooling1D";X(Y0);class Z0 extends X0{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),Vc(t,e,s,o,r,"avg")}}Z0.className="AveragePooling1D";X(Z0);class J0 extends It{constructor(t){if(t.poolSize==null&&(t.poolSize=[2,2]),super(t),this.poolSize=Array.isArray(t.poolSize)?t.poolSize:[t.poolSize,t.poolSize],t.strides==null)this.strides=this.poolSize;else if(Array.isArray(t.strides)){if(t.strides.length!==2)throw new F(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${t.strides.length}.`);this.strides=t.strides}else this.strides=[t.strides,t.strides];fe(this.poolSize,"poolSize"),fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,se(this.dataFormat),an(this.padding),this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){t=Nt(t);let e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2];return e=En(e,this.poolSize[0],this.padding,this.strides[0]),s=En(s,this.poolSize[1],this.padding,this.strides[1]),this.dataFormat==="channelsFirst"?[t[0],t[1],e,s]:[t[0],e,s,t[3]]}call(t,e){return z(()=>(this.invokeCallHook(t,e),this.poolingFunction(ft(t),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class Q0 extends J0{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),Vc(t,e,s,o,r,"max")}}Q0.className="MaxPooling2D";X(Q0);class ty extends J0{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),Vc(t,e,s,o,r,"avg")}}ty.className="AveragePooling2D";X(ty);class ey extends It{constructor(t){if(t.poolSize==null&&(t.poolSize=[2,2,2]),super(t),this.poolSize=Array.isArray(t.poolSize)?t.poolSize:[t.poolSize,t.poolSize,t.poolSize],t.strides==null)this.strides=this.poolSize;else if(Array.isArray(t.strides)){if(t.strides.length!==3)throw new F(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${t.strides.length}.`);this.strides=t.strides}else this.strides=[t.strides,t.strides,t.strides];fe(this.poolSize,"poolSize"),fe(this.strides,"strides"),this.padding=t.padding==null?"valid":t.padding,this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,se(this.dataFormat),an(this.padding),this.inputSpec=[new ae({ndim:5})]}computeOutputShape(t){t=Nt(t);let e=this.dataFormat==="channelsFirst"?t[2]:t[1],s=this.dataFormat==="channelsFirst"?t[3]:t[2],o=this.dataFormat==="channelsFirst"?t[4]:t[3];return e=En(e,this.poolSize[0],this.padding,this.strides[0]),s=En(s,this.poolSize[1],this.padding,this.strides[1]),o=En(o,this.poolSize[2],this.padding,this.strides[2]),this.dataFormat==="channelsFirst"?[t[0],t[1],e,s,o]:[t[0],e,s,o,t[4]]}call(t,e){return z(()=>(this.invokeCallHook(t,e),this.poolingFunction(ft(t),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const t={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class ny extends ey{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),K0(t,e,s,o,r,"max")}}ny.className="MaxPooling3D";X(ny);class sy extends ey{constructor(t){super(t)}poolingFunction(t,e,s,o,r){return se(r),an(o),K0(t,e,s,o,r,"avg")}}sy.className="AveragePooling3D";X(sy);class oy extends It{constructor(t){super(t),this.inputSpec=[new ae({ndim:3})]}computeOutputShape(t){return[t[0],t[2]]}call(t,e){throw new yt}}class ry extends oy{constructor(t){super(t||{})}call(t,e){return z(()=>{const s=ft(t);return te(s,1)})}}ry.className="GlobalAveragePooling1D";X(ry);class iy extends oy{constructor(t){super(t||{})}call(t,e){return z(()=>{const s=ft(t);return Nn(s,1)})}}iy.className="GlobalMaxPooling1D";X(iy);class ay extends It{constructor(t){super(t),this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,se(this.dataFormat),this.inputSpec=[new ae({ndim:4})]}computeOutputShape(t){return t=t,this.dataFormat==="channelsLast"?[t[0],t[3]]:[t[0],t[1]]}call(t,e){throw new yt}getConfig(){const t={dataFormat:this.dataFormat},e=super.getConfig();return Object.assign(t,e),t}}class ly extends ay{call(t,e){return z(()=>{const s=ft(t);return this.dataFormat==="channelsLast"?te(s,[1,2]):te(s,[2,3])})}}ly.className="GlobalAveragePooling2D";X(ly);class cy extends ay{call(t,e){return z(()=>{const s=ft(t);return this.dataFormat==="channelsLast"?Nn(s,[1,2]):Nn(s,[2,3])})}}cy.className="GlobalMaxPooling2D";X(cy);class uy extends It{constructor(t){super(t),this.layer=t.layer}build(t){this.built=!0}get trainable(){return this.layer!=null?this.layer.trainable:!1}set trainable(t){this.layer!=null&&(this.layer.trainable=t)}get trainableWeights(){return this.layer.trainableWeights}get nonTrainableWeights(){return this.layer.nonTrainableWeights}get updates(){return this.layer._updates}get losses(){return this.layer.losses}getWeights(){return this.layer.getWeights()}setWeights(t){this.layer.setWeights(t)}getConfig(){const t={layer:{className:this.layer.getClassName(),config:this.layer.getConfig()}},e=super.getConfig();return Object.assign(t,e),t}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.layer!=null&&this.layer.setFastWeightInitDuringBuild(t)}static fromConfig(t,e,s={}){const o=e.layer,r=Tn(o,s);delete e.layer;const i={layer:r};return Object.assign(i,e),new t(i)}}class hy extends uy{constructor(t){super(t),this.supportsMasking=!0}build(t){if(t=Nt(t),t.length<3)throw new F(`TimeDistributed layer expects an input shape >= 3D, but received input shape ${JSON.stringify(t)}`);this.inputSpec=[{shape:t}];const e=[t[0]].concat(t.slice(2));this.layer.built||(this.layer.build(e),this.layer.built=!0),super.build(t)}computeOutputShape(t){t=Nt(t);const e=[t[0]].concat(t.slice(2)),s=this.layer.computeOutputShape(e),o=t[1];return[s[0],o].concat(s.slice(1))}call(t,e){return z(()=>(t=ft(t),w0((i,a)=>[ft(this.layer.call(i,e)),[]],t,[],!1,null,null,!1,!0)[1]))}}hy.className="TimeDistributed";X(hy);function YA(n){To(gR,"BidirectionalMergeMode",n)}const ZA="concat";class dy extends uy{constructor(t){super(t);const e=t.layer.getConfig(),s={};s.className=t.layer.getClassName(),s.config=e,this.forwardLayer=Tn(s),e.goBackwards=e.goBackwards!==!0;const o={};if(o.className=t.layer.getClassName(),o.config=e,this.backwardLayer=Tn(o),this.forwardLayer.name="forward_"+this.forwardLayer.name,this.backwardLayer.name="backward_"+this.backwardLayer.name,this.mergeMode=t.mergeMode===void 0?ZA:t.mergeMode,YA(this.mergeMode),t.weights)throw new yt("weights support is not implemented for Bidirectional layer yet.");this._stateful=t.layer.stateful,this.returnSequences=t.layer.returnSequences,this.returnState=t.layer.returnState,this.supportsMasking=!0,this._trainable=!0,this.inputSpec=t.layer.inputSpec,this.numConstants=null}get trainable(){return this._trainable}set trainable(t){this._trainable=t,this.forwardLayer!=null&&(this.forwardLayer.trainable=t),this.backwardLayer!=null&&(this.backwardLayer.trainable=t)}getWeights(){return this.forwardLayer.getWeights().concat(this.backwardLayer.getWeights())}setWeights(t){const e=t.length,s=Math.floor(e/2);this.forwardLayer.setWeights(t.slice(0,s)),this.backwardLayer.setWeights(t.slice(s))}computeOutputShape(t){let e=this.forwardLayer.computeOutputShape(t);Array.isArray(e)&&Array.isArray(e[0])||(e=[e]),e=e;let s,o,r;return this.returnState&&(r=e.slice(1)),s=e[0],s=s,this.mergeMode==="concat"?(s[s.length-1]*=2,o=[s]):this.mergeMode==null?o=[s,s.slice()]:o=[s],this.returnState?this.mergeMode==null?o.concat(r).concat(r.slice()):[s].concat(r).concat(r.slice()):Pe(o)}apply(t,e){let s=e==null?null:e.initialState,o=e==null?null:e.constants;e==null&&(e={});const r=y0(t,s,o,this.numConstants);if(t=r.inputs,s=r.initialState,o=r.constants,Array.isArray(t)&&(s=t.slice(1),t=t[0]),(s==null||s.length===0)&&o==null)return super.apply(t,e);const i=[],a=[];if(s!=null){const c=s.length;if(c%2>0)throw new F("When passing `initialState` to a Bidrectional RNN, the state should be an Array containing the states of the underlying RNNs.");e.initialState=s,i.push(...s);const u=s.map(h=>new ae({shape:h.shape}));this.forwardLayer.stateSpec=u.slice(0,c/2),this.backwardLayer.stateSpec=u.slice(c/2),a.push(...u)}if(o!=null)throw new yt("Support for constants in Bidirectional layers is not implemented yet.");const l=i[0]instanceof qn;for(const c of i)if(c instanceof qn!==l)throw new F("The initial state of a Bidirectional layer cannot be specified as a mix of symbolic and non-symbolic tensors");if(l){const c=[t].concat(i),u=this.inputSpec.concat(a),h=this.inputSpec;this.inputSpec=u;const d=super.apply(c,e);return this.inputSpec=h,d}else return super.apply(t,e)}call(t,e){return z(()=>{const s=e.initialState;let o,r;if(s==null)o=this.forwardLayer.call(t,e),r=this.backwardLayer.call(t,e);else{const l=s.slice(0,s.length/2),c=s.slice(s.length/2);o=this.forwardLayer.call(t,Object.assign(e,{initialState:l})),r=this.backwardLayer.call(t,Object.assign(e,{initialState:c}))}let i;this.returnState&&(Array.isArray(o)&&(i=o.slice(1).concat(r.slice(1))),o=o[0],r=r[0]),this.returnSequences&&(r=go(r,1));let a;return this.mergeMode==="concat"?a=fp([o,r]):this.mergeMode==="sum"?a=J(o,r):this.mergeMode==="ave"?a=A(.5,J(o,r)):this.mergeMode==="mul"?a=A(o,r):this.mergeMode==null&&(a=[o,r]),this.returnState?this.mergeMode==null?a.concat(i):[a].concat(i):a})}resetStates(t){this.forwardLayer.resetStates(),this.backwardLayer.resetStates()}build(t){no(this.forwardLayer.name,()=>{this.forwardLayer.build(t)}),no(this.backwardLayer.name,()=>{this.backwardLayer.build(t)}),this.built=!0}computeMask(t,e){Array.isArray(e)&&(e=e[0]);let s;if(this.returnSequences?this.mergeMode==null?s=[e,e]:s=e:this.mergeMode==null?s=[null,null]:s=null,this.returnState){const r=this.forwardLayer.states.map(i=>null);return Array.isArray(s)?s.concat(r).concat(r):[s].concat(r).concat(r)}else return s}get trainableWeights(){return this.forwardLayer.trainableWeights.concat(this.backwardLayer.trainableWeights)}get nonTrainableWeights(){return this.forwardLayer.nonTrainableWeights.concat(this.backwardLayer.nonTrainableWeights)}setFastWeightInitDuringBuild(t){super.setFastWeightInitDuringBuild(t),this.forwardLayer!=null&&this.forwardLayer.setFastWeightInitDuringBuild(t),this.backwardLayer!=null&&this.backwardLayer.setFastWeightInitDuringBuild(t)}getConfig(){const t={mergeMode:this.mergeMode},e=super.getConfig();return Object.assign(t,e),t}static fromConfig(t,e){const s=Tn(e.layer);if(delete e.layer,e.numConstants!=null)throw new yt("Deserialization of a Bidirectional layer with numConstants present is not supported yet.");const o=e;return o.layer=s,new t(o)}}dy.className="Bidirectional";X(dy);class py extends It{constructor(t){super(t),this.scale=t.scale,t.offset?this.offset=t.offset:this.offset=0}getConfig(){const t={scale:this.scale,offset:this.offset},e=super.getConfig();return Object.assign(t,e),t}call(t,e){return z(()=>(t=ft(t),t.dtype!=="float32"&&(t=Wn(t,"float32")),J(A(t,this.scale),this.offset)))}}py.className="Rescaling";X(py);const{resizeBilinear:JA,cropAndResize:QA}=ts;class fy extends It{constructor(t){super(t),this.height=t.height,this.width=t.width}centerCrop(t,e,s,o,r,i,a,l){return z(()=>{let c,u=!1;const h=e/i,d=s/a,p=(o+e)/i,f=(r+s)/a,m=[h,d,p,f],g=[];t.rank===3?(u=!0,c=Hn([t])):c=t;for(let $=0;$<c.shape[0];$++)g.push(m);const x=Bo(g,[g.length,4]),b=Mr(0,g.length,1,"int32"),y=QA(c,x,b,[o,r],"nearest");return Wn(u?ft(xo(y)):y,l)})}upsize(t,e,s,o){return z(()=>{const r=JA(t,[e,s]);return Wn(r,o)})}call(t,e){return z(()=>{const s=ft(t),o=s.dtype,r=s.shape,i=r[r.length-3],a=r[r.length-2];let l=0;i!==this.height&&(l=Math.floor((i-this.height)/2));let c=0;return a!==this.width&&(c=Math.floor((a-this.width)/2),c===0&&(c=1)),l>=0&&c>=0?this.centerCrop(s,l,c,this.height,this.width,i,a,o):this.upsize(t,this.height,this.width,o)})}getConfig(){const t={height:this.height,width:this.width},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){t=Nt(t);const e=t.length-3,s=t.length-2;return t[e]=this.height,t[s]=this.width,t}}fy.className="CenterCrop";X(fy);function tD(n,t,e,s){let o=ft(n);if(o.dtype!=="int32"&&(o=Wn(o,"int32")),t==="int")return o;const r=o.shape;if(o.rank===0&&(o=Me(o,-1)),t==="oneHot"&&o.shape[o.shape.length-1]!==1&&(o=Me(o,-1)),o.rank>2)throw new F(`When outputMode is not int, maximum output rank is 2 Received outputMode ${t} and input shape ${r} which would result in output rank ${o.rank}.`);const i=["multiHot","oneHot"].includes(t),a=o;let l;if(typeof s<"u"&&t==="count"?l=wf(a,s,e,i):l=wf(a,[],e,i),t!=="tfIdf")return l;if(s)return A(l,s);throw new F("When outputMode is 'tfIdf', weights must be provided.")}class my extends It{constructor(t){super(t),this.numTokens=t.numTokens,t.outputMode?this.outputMode=t.outputMode:this.outputMode="multiHot"}getConfig(){const t={numTokens:this.numTokens,outputMode:this.outputMode},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){return t=Nt(t),t==null?[this.numTokens]:this.outputMode==="oneHot"&&t[t.length-1]!==1?(t.push(this.numTokens),t):(t[t.length-1]=this.numTokens,t)}call(t,e){return z(()=>{t=ft(t),t.dtype!=="int32"&&(t=Wn(t,"int32"));let s;if(typeof e.countWeights<"u"){if(this.outputMode!=="count")throw new F(`countWeights is not used when outputMode !== count.
              Received countWeights=${e.countWeights}`);s=ft(e.countWeights)}const o=Nn(t),r=Ha(t),i=Ze(this.numTokens,o).bufferSync().get(0),a=So(r,0).bufferSync().get(0);if(!(i&&a))throw new F(`Input values must be between 0 < values <= numTokens with numTokens=${this.numTokens}`);return tD(t,this.outputMode,this.numTokens,s)})}}my.className="CategoryEncoding";X(my);const eD=["bilinear","nearest"],cm=new Set(eD);class gy extends It{constructor(t){if(super(t),this.height=t.height,this.width=t.width,t.interpolation)if(cm.has(t.interpolation))this.interpolation=t.interpolation;else throw new F(`Invalid interpolation parameter: ${t.interpolation} is not implemented`);else this.interpolation="bilinear";this.cropToAspectRatio=!!t.cropToAspectRatio}computeOutputShape(t){t=Nt(t);const e=t[2];return[this.height,this.width,e]}getConfig(){const t={height:this.height,width:this.width,interpolation:this.interpolation,cropToAspectRatio:this.cropToAspectRatio},e=super.getConfig();return Object.assign(t,e),t}call(t,e){return z(()=>{const s=[this.height,this.width];if(this.interpolation==="bilinear")return ts.resizeBilinear(t,s,!this.cropToAspectRatio);if(this.interpolation==="nearest")return ts.resizeNearestNeighbor(t,s,!this.cropToAspectRatio);throw new Error(`Interpolation is ${this.interpolation} but only ${[...cm]} are supported`)})}}gy.className="Resizing";X(gy);class xy{constructor(t){this.seed=t}next(){if(this.seed!==void 0)return this.seed++}}xy.className="RandomSeed";class by extends It{constructor(t){super(t),this.randomGenerator=new xy(t.seed)}getConfig(){const t={seed:this.randomGenerator.seed},e=super.getConfig();return Object.assign(t,e),t}}by.className="BaseRandomLayer";const nD=["bilinear","nearest"],um=new Set(nD);class yy extends by{constructor(t){super(t);const{factor:e,interpolation:s="bilinear"}=t;if(this.factor=e,Array.isArray(this.factor)&&this.factor.length===2)this.widthLower=this.factor[0],this.widthUpper=this.factor[1];else if(!Array.isArray(this.factor)&&this.factor>0)this.widthLower=-this.factor,this.widthUpper=this.factor;else throw new F(`Invalid factor: ${this.factor}. Must be positive number or tuple of 2 numbers`);if(this.widthLower<-1||this.widthUpper<-1)throw new F(`factor must have values larger than -1. Got: ${this.factor}`);if(this.widthUpper<this.widthLower)throw new F(`factor cannot have upper bound less than lower bound.
        Got upper bound: ${this.widthUpper}.
        Got lower bound: ${this.widthLower}
      `);if(s)if(um.has(s))this.interpolation=s;else throw new F(`Invalid interpolation parameter: ${s} is not implemented`)}getConfig(){const t={factor:this.factor,interpolation:this.interpolation},e=super.getConfig();return Object.assign(t,e),t}computeOutputShape(t){t=Nt(t);const e=t[2];return[this.imgHeight,-1,e]}call(t,e){return z(()=>{const s=ft(t);this.imgHeight=s.shape[s.shape.length-3];const o=s.shape[s.shape.length-2];this.widthFactor=Ki([1],1+this.widthLower,1+this.widthUpper,"float32",this.randomGenerator.next());let r=this.widthFactor.dataSync()[0]*o;r=Math.round(r);const i=[this.imgHeight,r];switch(this.interpolation){case"bilinear":return ts.resizeBilinear(t,i);case"nearest":return ts.resizeNearestNeighbor(t,i);default:throw new Error(`Interpolation is ${this.interpolation}
          but only ${[...um]} are supported`)}})}}yy.className="RandomWidth";X(yy);const sD=B();sD.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,n=>{n&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});var hm;(function(n){n[n.DT_INVALID=0]="DT_INVALID",n[n.DT_FLOAT=1]="DT_FLOAT",n[n.DT_DOUBLE=2]="DT_DOUBLE",n[n.DT_INT32=3]="DT_INT32",n[n.DT_UINT8=4]="DT_UINT8",n[n.DT_INT16=5]="DT_INT16",n[n.DT_INT8=6]="DT_INT8",n[n.DT_STRING=7]="DT_STRING",n[n.DT_COMPLEX64=8]="DT_COMPLEX64",n[n.DT_INT64=9]="DT_INT64",n[n.DT_BOOL=10]="DT_BOOL",n[n.DT_QINT8=11]="DT_QINT8",n[n.DT_QUINT8=12]="DT_QUINT8",n[n.DT_QINT32=13]="DT_QINT32",n[n.DT_BFLOAT16=14]="DT_BFLOAT16",n[n.DT_QINT16=15]="DT_QINT16",n[n.DT_QUINT16=16]="DT_QUINT16",n[n.DT_UINT16=17]="DT_UINT16",n[n.DT_COMPLEX128=18]="DT_COMPLEX128",n[n.DT_HALF=19]="DT_HALF",n[n.DT_RESOURCE=20]="DT_RESOURCE",n[n.DT_VARIANT=21]="DT_VARIANT",n[n.DT_UINT32=22]="DT_UINT32",n[n.DT_UINT64=23]="DT_UINT64",n[n.DT_FLOAT_REF=101]="DT_FLOAT_REF",n[n.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",n[n.DT_INT32_REF=103]="DT_INT32_REF",n[n.DT_UINT8_REF=104]="DT_UINT8_REF",n[n.DT_INT16_REF=105]="DT_INT16_REF",n[n.DT_INT8_REF=106]="DT_INT8_REF",n[n.DT_STRING_REF=107]="DT_STRING_REF",n[n.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",n[n.DT_INT64_REF=109]="DT_INT64_REF",n[n.DT_BOOL_REF=110]="DT_BOOL_REF",n[n.DT_QINT8_REF=111]="DT_QINT8_REF",n[n.DT_QUINT8_REF=112]="DT_QUINT8_REF",n[n.DT_QINT32_REF=113]="DT_QINT32_REF",n[n.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",n[n.DT_QINT16_REF=115]="DT_QINT16_REF",n[n.DT_QUINT16_REF=116]="DT_QUINT16_REF",n[n.DT_UINT16_REF=117]="DT_UINT16_REF",n[n.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",n[n.DT_HALF_REF=119]="DT_HALF_REF",n[n.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",n[n.DT_VARIANT_REF=121]="DT_VARIANT_REF",n[n.DT_UINT32_REF=122]="DT_UINT32_REF",n[n.DT_UINT64_REF=123]="DT_UINT64_REF"})(hm||(hm={}));var dm;(function(n){(function(t){t[t.LEGACY=0]="LEGACY",t[t.V1=1]="V1",t[t.V2=2]="V2"})(n.CheckpointFormatVersion||(n.CheckpointFormatVersion={}))})(dm||(dm={}));function oD(n,t){return rl(n,t)}function rl(n,t,e=new Map,s=new Set){if(n==null)return null;if(typeof Blob=="function"&&n instanceof Blob)return n.slice();if(s.has(n))throw new Error("Circular references are not supported.");if(e.has(n))return e.get(n);const o=t(n);if(o.recurse&&o.value!==null)throw new Error("A deep map function may not return both a value and recurse=true.");if(o.recurse)if(Xo(n)){const r=Array.isArray(n)?[]:{};s.add(n);for(const i in n){const a=n[i],l=rl(a,t,e,s);r[i]=l}return s.delete(n),n.__proto__&&(r.__proto__=n.__proto__),r}else throw new Error(`Can't recurse into non-iterable type: ${n}`);else return e.set(n,o.value),o.value}function rD(n,t=$y){return wy(n,t)}function wy(n,t,e=new Set){const s=n[0];if(e.has(s))throw new Error("Circular references are not supported.");const o=t(n);if(o.recurse&&o.value!==null)throw new Error("A deep zip function may not return both a value and recurse=true.");if(o.recurse)if(Xo(s)){const r=Array.isArray(s)?[]:{};e.add(s);for(const i in s){const a=n.map(c=>c[i]),l=wy(a,t,e);r[i]=l}return e.delete(s),r}else throw new Error(`Can't recurse into non-iterable type: ${s}`);else return o.value}function $y(n){return n===null?null:Xo(n[0])?{value:null,recurse:!0}:{value:n,recurse:!1}}async function Cy(n,t){const e=new Map;rl(n,t,e);for(const o of Array.from(e.keys())){const r=e.get(o);if(Kr(r)){const i=await r;e.set(o,i)}}return rl(n,t,e)}function Xo(n){let t=!1;if(B().get("IS_BROWSER"))t=n instanceof TextDecoder;else{const{StringDecoder:e}=require("string_decoder");t=n instanceof e}return n!=null&&!ArrayBuffer.isView(n)&&(Array.isArray(n)||typeof n=="object"&&!(n instanceof Lt)&&!(n instanceof Promise)&&!t)}function iD(n){return n==null||aD(n)||Array.isArray(n)||typeof n=="object"&&n instanceof Lt||je(n)}function aD(n){return n===null||typeof n!="object"&&typeof n!="function"}function lD(n){return oD(n,cD)}function cD(n){return n instanceof Lt?{value:n.clone(),recurse:!1}:Xo(n)?{value:null,recurse:!0}:{value:n,recurse:!1}}class Iy{constructor(t){if(this.capacity=t,this.begin=0,this.end=0,t==null)throw new RangeError("Can't create a ring buffer of unknown capacity.");if(t<1)throw new RangeError("Can't create ring buffer of capacity < 1.");this.data=new Array(t),this.doubledCapacity=2*t}wrap(t){for(;t<0;)t+=this.doubledCapacity;return t%this.doubledCapacity}get(t){if(t<0)throw new RangeError("Can't get item at a negative index.");return this.data[t%this.capacity]}set(t,e){if(t<0)throw new RangeError("Can't set item at a negative index.");this.data[t%this.capacity]=e}length(){let t=this.end-this.begin;return t<0&&(t=this.doubledCapacity+t),t}isFull(){return this.length()===this.capacity}isEmpty(){return this.length()===0}push(t){if(this.isFull())throw new RangeError("Ring buffer is full.");this.set(this.end,t),this.end=this.wrap(this.end+1)}pushAll(t){for(const e of t)this.push(e)}pop(){if(this.isEmpty())throw new RangeError("Ring buffer is empty.");this.end=this.wrap(this.end-1);const t=this.get(this.end);return this.set(this.end,void 0),t}unshift(t){if(this.isFull())throw new RangeError("Ring buffer is full.");this.begin=this.wrap(this.begin-1),this.set(this.begin,t)}shift(){if(this.isEmpty())throw new RangeError("Ring buffer is empty.");const t=this.get(this.begin);return this.set(this.begin,void 0),this.begin=this.wrap(this.begin+1),t}shuffleExcise(t){if(this.isEmpty())throw new RangeError("Ring buffer is empty.");const e=this.wrap(this.begin+t),s=this.get(e);return this.set(e,this.pop()),s}}class Wc extends Iy{constructor(){super(Wc.INITIAL_CAPACITY)}isFull(){return!1}push(t){super.isFull()&&this.expand(),super.push(t)}unshift(t){super.isFull()&&this.expand(),super.unshift(t)}expand(){const t=this.capacity*2,e=new Array(t),s=this.length();for(let o=0;o<s;o++)e[o]=this.get(this.wrap(this.begin+o));this.data=e,this.capacity=t,this.doubledCapacity=2*this.capacity,this.begin=0,this.end=s}}Wc.INITIAL_CAPACITY=32;function vy(n){return new pD(n)}function uD(n){return new fD(n)}function hD(n,t){return new ky(n,t)}function dD(n,t=Cs.FAIL){return new vD(n,t)}class We{async toArray(){const t=[];let e=await this.next();for(;!e.done;)t.push(e.value),e=await this.next();return t}async toArrayForTest(){const t=this.prefetch(100),e=[];let s=await t.next();for(;!s.done;)e.push(s.value),s=await t.next();return e}async resolveFully(){let t=await this.next();for(;!t.done;)t=await this.next()}async resolveWhile(t){let e=await this.next(),s=t(e.value);for(;!e.done&&s;)e=await this.next(),s=t(e.value)}handleErrors(t){return new $D(this,t)}filter(t){return new yD(this,t)}map(t){return new wD(this,t)}mapAsync(t){return new pm(this,t)}serialMapAsync(t){return new pm(this,t).serial()}flatmap(t){return new ID(this,t)}async forEachAsync(t){return this.map(t).resolveFully()}async serialForEach(t){return this.serialMapAsync(t).resolveWhile(e=>e===!0)}rowMajorBatch(t,e=!0){return new bD(this,t,e)}columnMajorBatch(t,e=!0,s=$y){return this.rowMajorBatch(t,e).map(r=>rD(r,s))}concatenate(t,e){return new ky(vy([this,t]),e)}take(t){return t<0||t==null?this:new xD(this,t)}skip(t){return t<0||t==null?this:new gD(this,t)}prefetch(t){return new Sy(this,t)}shuffle(t,e){return new kD(this,t,e)}serial(){return new mD(this)}}class pD extends We{constructor(t){super(),this.items=t,this.trav=0}summary(){return`Array of ${this.items.length} items`}async next(){if(this.trav>=this.items.length)return{value:null,done:!0};const t=this.items[this.trav];return this.trav++,{value:lD(t),done:!1}}}class fD extends We{constructor(t){super(),this.nextFn=t}summary(){return"Function call"}async next(){try{return this.nextFn()}catch(t){throw t.message=`Error thrown while iterating through a dataset: ${t.message}`,t}}}class mD extends We{constructor(t){super(),this.upstream=t,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Serial`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){return this.upstream.next()}}class gD extends We{constructor(t,e){super(),this.upstream=t,this.maxCount=e,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Skip`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;this.count++<this.maxCount;){const t=await this.upstream.next();if(t.done)return t;xt(t.value)}return this.upstream.next()}}class xD extends We{constructor(t,e){super(),this.upstream=t,this.maxCount=e,this.count=0}summary(){return`${this.upstream.summary()} -> Take`}async next(){return this.count++>=this.maxCount?{value:null,done:!0}:this.upstream.next()}}class bD extends We{constructor(t,e,s=!0){super(),this.upstream=t,this.batchSize=e,this.enableSmallLastBatch=s,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> RowMajorBatch`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){const t=[];for(;t.length<this.batchSize;){const e=await this.upstream.next();if(e.done)return this.enableSmallLastBatch&&t.length>0?{value:t,done:!1}:{value:null,done:!0};t.push(e.value)}return{value:t,done:!1}}}class yD extends We{constructor(t,e){super(),this.upstream=t,this.predicate=e,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> Filter`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;){const t=await this.upstream.next();if(t.done||this.predicate(t.value))return t;xt(t.value)}}}class wD extends We{constructor(t,e){super(),this.upstream=t,this.transform=e}summary(){return`${this.upstream.summary()} -> Map`}async next(){const t=await this.upstream.next();if(t.done)return{value:null,done:!0};const e=ss(t.value),s=this.transform(t.value),o=ss(s);for(const r of e)xc(r,o)||r.dispose();return{value:s,done:!1}}}class $D extends We{constructor(t,e){super(),this.upstream=t,this.handler=e,this.count=0,this.lastRead=Promise.resolve({value:null,done:!1})}summary(){return`${this.upstream.summary()} -> handleErrors`}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;;)try{return await this.upstream.next()}catch(t){if(!this.handler(t))return{value:null,done:!0}}}}class pm extends We{constructor(t,e){super(),this.upstream=t,this.transform=e}summary(){return`${this.upstream.summary()} -> AsyncMap`}async next(){const t=await this.upstream.next();if(t.done)return{value:null,done:!0};const e=ss(t.value),s=await this.transform(t.value),o=ss(s);for(const r of e)xc(r,o)||r.dispose();return{value:s,done:!1}}}class CD extends We{constructor(){super(),this.outputQueue=new Wc,this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}async serialNext(){for(;this.outputQueue.length()===0;)if(!await this.pump())return{value:null,done:!0};return{value:this.outputQueue.shift(),done:!1}}}class ID extends CD{constructor(t,e){super(),this.upstream=t,this.transform=e}summary(){return`${this.upstream.summary()} -> Flatmap`}async pump(){const t=await this.upstream.next();if(t.done)return!1;const e=ss(t.value),s=this.transform(t.value),o=ss(s);this.outputQueue.pushAll(s);for(const r of e)xc(r,o)||r.dispose();return!0}}class ky extends We{constructor(t,e){super(),this.baseErrorHandler=e,this.lastRead=null,this.iterator=null,this.moreIterators=t}summary(){return"TODO: fill in upstream of chained summaries -> Chained"}async next(){return this.lastRead=this.readFromChain(this.lastRead),this.lastRead}async readFromChain(t){if(await t,this.iterator==null){const s=await this.moreIterators.next();if(s.done)return{value:null,done:!0};this.iterator=s.value,this.baseErrorHandler!=null&&(this.iterator=this.iterator.handleErrors(this.baseErrorHandler))}const e=await this.iterator.next();return e.done?(this.iterator=null,this.readFromChain(t)):e}}var Cs;(function(n){n[n.FAIL=0]="FAIL",n[n.SHORTEST=1]="SHORTEST",n[n.LONGEST=2]="LONGEST"})(Cs||(Cs={}));class vD extends We{constructor(t,e=Cs.FAIL){super(),this.iterators=t,this.mismatchMode=e,this.count=0,this.currentPromise=null}summary(){return"{TODO: fill in upstream of zip summaries} -> Zip"}async nextState(t){await t;let e=0,s=0;function o(i){return i instanceof We?{value:i.next().then(l=>(e++,l.done&&s++,l.value)),recurse:!1}:{value:null,recurse:!0}}const r=await Cy(this.iterators,o);if(e===s)return{value:null,done:!0};if(s>0)switch(this.mismatchMode){case Cs.FAIL:throw new Error(`Zipped streams should have the same length. Mismatched at element ${this.count}.`);case Cs.SHORTEST:return{value:null,done:!0};case Cs.LONGEST:}return this.count++,{value:r,done:!1}}async next(){return this.currentPromise=this.nextState(this.currentPromise),this.currentPromise}}class Sy extends We{constructor(t,e){super(),this.upstream=t,this.bufferSize=e,this.buffer=new Iy(e)}summary(){return`${this.upstream.summary()} -> Prefetch`}refill(){for(;!this.buffer.isFull();){const t=this.upstream.next();this.buffer.push(t)}}next(){return this.refill(),this.buffer.shift()}}class kD extends Sy{constructor(t,e,s){super(t,e),this.upstream=t,this.windowSize=e,this.upstreamExhausted=!1,this.random=sr.alea(s||xe().toString()),this.lastRead=Promise.resolve({value:null,done:!1})}async next(){return this.lastRead=this.lastRead.then(()=>this.serialNext()),this.lastRead}randomInt(t){return Math.floor(this.random()*t)}chooseIndex(){return this.randomInt(this.buffer.length())}async serialNext(){for(this.upstreamExhausted||this.refill();!this.buffer.isEmpty();){const t=this.chooseIndex(),e=await this.buffer.shuffleExcise(t);if(e.done)this.upstreamExhausted=!0;else return this.refill(),e}return{value:null,done:!0}}}class _p{constructor(){this.size=null}batch(t,e=!0){const s=this;I(t>0,()=>`batchSize needs to be positive, but it is
      ${t}`);let o;return this.size===1/0||this.size==null?o=this.size:e?o=Math.ceil(this.size/t):o=Math.floor(this.size/t),un(async()=>(await s.iterator()).columnMajorBatch(t,e,SD),o)}concatenate(t){const e=this;let s;return this.size===1/0||t.size===1/0?s=1/0:this.size!=null&&t.size!=null?s=this.size+t.size:s=null,un(async()=>(await e.iterator()).concatenate(await t.iterator()),s)}filter(t){const e=this;let s;return this.size===1/0?s=1/0:s=null,un(async()=>(await e.iterator()).filter(o=>z(()=>t(o))),s)}async forEachAsync(t){return(await this.iterator()).forEachAsync(t)}map(t){const e=this;return un(async()=>(await e.iterator()).map(s=>z(()=>t(s))),this.size)}mapAsync(t){const e=this;return un(async()=>(await e.iterator()).mapAsync(t),this.size)}prefetch(t){if(t==null)throw new RangeError("`Dataset.prefetch()` requires bufferSize to be specified.");const e=this;return un(async()=>(await e.iterator()).prefetch(t),this.size)}repeat(t){const e=this;let s;return this.size!=null&&t>0?s=this.size*t:t===0?s=0:this.size!=null&&(t===void 0||t<0)?s=1/0:s=null,un(async()=>{const o=uD(async()=>({value:await e.iterator(),done:!1}));return hD(o.take(t))},s)}skip(t){const e=this;let s;return this.size!=null&&t>=0&&this.size>=t?s=this.size-t:this.size!=null&&(this.size<t||t===void 0||t<0)?s=0:s=null,un(async()=>(await e.iterator()).skip(t),s)}shuffle(t,e,s=!0){if(t==null||t<0)throw this.size==null?new RangeError("`Dataset.shuffle()` requires bufferSize to be specified."):new RangeError(`\`Dataset.shuffle()\` requires bufferSize to be specified.  If your data fits in main memory (for regular JS objects), and/or GPU memory (for \`tf.Tensor\`s), consider setting bufferSize to the dataset size (${this.size} elements)`);const o=this,r=sr.alea(e||xe().toString());return un(async()=>{let i=r.int32();return s&&(i+=r.int32()),(await o.iterator()).shuffle(t,i.toString())},this.size)}take(t){const e=this;let s;return this.size!=null&&this.size>t?s=t:this.size!=null&&this.size<=t?s=this.size:s=null,un(async()=>(await e.iterator()).take(t),s)}async toArray(){if(this.size===1/0)throw new Error("Can not convert infinite data stream to array.");return(await this.iterator()).toArray()}async toArrayForTest(){if(this.size===1/0)throw new Error("Can not convert infinite data stream to array.");return(await this.iterator()).toArrayForTest()}}_p.MAX_BUFFER_SIZE=1e4;function un(n,t=null){return new class extends _p{constructor(){super(...arguments),this.size=t}async iterator(){return n()}}}function b6(n){return un(async()=>vy(n),n.length)}function y6(n){if(!Xo(n))throw new Error("The argument to zip() must be an object or array.");let t;if(Array.isArray(n))for(let e=0;e<n.length;e++)t=t==null?n[e].size:Math.min(t,n[e].size);else if(n instanceof Object)for(const e in n)t=t==null?n[e].size:Math.min(t,n[e].size);return un(async()=>{const e=await Cy(n,s=>{if(s instanceof _p)return{value:s.iterator(),recurse:!1};if(Xo(s))return{value:null,recurse:!0};throw new Error("Leaves of the structure passed to zip() must be Datasets, not primitives.")});return dD(e,Cs.SHORTEST)},t)}function SD(n){if(n===null)return null;const t=n[0];return iD(t)?{value:ND(n),recurse:!1}:{value:null,recurse:!0}}function ND(n){if(n.length===0)throw new Error("Can't make a batch of zero elements.");return n[0]instanceof Lt?Hn(n):Bo(n)}function at(n,t){Array.isArray(n)||(n=[n]),n.forEach(e=>{e!=null&&I(e.dtype!=="complex64",()=>`${t} does not support complex64 tensors in the CPU backend.`)})}const TD=cx;class Uc extends Yu{nextDataId(){return Uc.nextDataId++}constructor(){super(),this.blockSize=48,this.firstUse=!0,this.data=new Gm(this,On())}write(t,e,s){this.firstUse&&(this.firstUse=!1,B().get("IS_NODE")&&Qe(`
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================`));const o={id:this.nextDataId()};return this.data.set(o,{values:t,dtype:s,refCount:1}),o}makeTensorInfo(t,e,s){let o;if(e==="string"&&s!=null&&s.length>0&&Qo(s[0])){const r=s.map(i=>es(i));o=this.write(r,t,e)}else o=this.write(s,t,e);return{dataId:o,shape:t,dtype:e}}refCount(t){return this.data.has(t)?this.data.get(t).refCount:0}incRef(t){const e=this.data.get(t);e.refCount++}decRef(t){if(this.data.has(t)){const e=this.data.get(t);e.refCount--}}move(t,e,s,o,r){this.data.set(t,{values:e,dtype:o,refCount:r})}numDataIds(){return this.data.numDataIds()}async read(t){return this.readSync(t)}readSync(t){const{dtype:e,complexTensorInfos:s}=this.data.get(t);if(e==="complex64"){const o=this.readSync(s.real.dataId),r=this.readSync(s.imag.dataId);return os(o,r)}return Zm(this.data.get(t).values,e)}bufferSync(t){const e=this.readSync(t.dataId);if(t.dtype==="string")try{const s=e.map(o=>ns(o));return Ct(t.shape,t.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return Ct(t.shape,t.dtype,e)}makeOutput(t,e,s){return On().makeTensorFromTensorInfo(this.makeTensorInfo(e,s,t),this)}disposeData(t,e=!1){if(this.data.has(t)){if(this.data.get(t).refCount--,!e&&this.data.get(t).refCount>0)return!1;const{complexTensorInfos:s}=this.data.get(t);s!=null&&(this.disposeData(s.real.dataId,!0),this.disposeData(s.imag.dataId,!0)),this.data.delete(t)}return!0}disposeIntermediateTensorInfo(t){this.disposeData(t.dataId)}async time(t){const e=xe();return t(),{kernelMs:xe()-e}}memory(){return{unreliable:!0,reasons:["The reported memory is an upper bound. Due to automatic garbage collection, the true allocated memory may be less."]}}where(t){at([t],"where");const e=this.readSync(t.dataId);return TD(t.shape,e)}dispose(){}floatPrecision(){return 32}epsilon(){return super.epsilon()}}Uc.nextDataId=0;function Ny(n){const t=new Float32Array(n.length);for(let e=0;e<n.length;++e)t[e]=Math.abs(n[e]);return t}const ED=n=>{const{x:t}=n.inputs,e=n.backend;at(t,"abs");let s=new Float32Array(U(t.shape));const o=e.data.get(t.dataId).values;return s=Ny(o),e.makeOutput(s,t.shape,t.dtype)},RD={kernelName:gl,backendName:"cpu",kernelFunc:ED};function oe(n){return(t,e,s,o,r)=>{const i=gt(t,e),a=i.length,l=ut(i),c=U(i),u=we(r,c),h=t.length,d=e.length,p=ut(t),f=ut(e),m=po(t,i),g=po(e,i);if(m.length+g.length===0)for(let x=0;x<u.length;++x)u[x]=n(s[x%s.length],o[x%o.length]);else for(let x=0;x<u.length;++x){const b=ko(x,a,l),w=b.slice(-h);m.forEach(N=>w[N]=0);const y=Sn(w,h,p),$=b.slice(-d);g.forEach(N=>$[N]=0);const v=Sn($,d,f);u[x]=n(s[y],o[v])}return[u,i]}}function qe(n){const{inputs:t,backend:e}=n,{real:s,imag:o}=t,r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,a=e.makeTensorInfo(s.shape,"complex64"),l=e.data.get(a.dataId);return l.complexTensorInfos={real:e.makeTensorInfo(s.shape,"float32",r),imag:e.makeTensorInfo(o.shape,"float32",i)},a}const AD={kernelName:uh,backendName:"cpu",kernelFunc:qe};function il(n,t,e="float32"){if(e==="complex64"){const o=il(n,t,"float32"),r=il(n,t,"float32");return qe({inputs:{real:o,imag:r},backend:n})}const s=$e(U(t),e);return n.makeTensorInfo(t,e,s)}function jn(n){const{inputs:t,backend:e}=n,{x:s}=t;return e.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const DD={kernelName:mi,backendName:"cpu",kernelFunc:jn};function yo(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.data.get(s.dataId).complexTensorInfos.real,r=e.data.get(o.dataId).values;return e.makeTensorInfo(o.shape,o.dtype,r)}const FD={kernelName:Oh,backendName:"cpu",kernelFunc:yo};function Ty(n,t,e,s){if(s==="int32"){const o=Int32Array.from(n);return[t,"int32",o]}if(s==="bool"){const o=Ls([0],e),[r,i]=oe((a,l)=>a!==l?1:0)(t,[],n,o,"bool");return[i,"bool",r]}throw new Error(`Error in Cast: failed to cast ${e} to ${s}`)}function Os(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return jn({inputs:{x:o},backend:e});const u=il(e,o.shape,o.dtype),h=Os({inputs:{x:o},backend:e,attrs:{dtype:"float32"}}),d=qe({inputs:{real:h,imag:u},backend:e});return e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),d}if(o.dtype==="complex64"){const u=yo({inputs:{input:o},backend:e}),h=Os({inputs:{x:u},backend:e,attrs:{dtype:r}});return e.disposeIntermediateTensorInfo(u),h}if(!th(o.dtype,r)){const u=jn({inputs:{x:o},backend:e});return{dataId:u.dataId,shape:u.shape,dtype:r}}const i=e.data.get(o.dataId).values,[a,l,c]=Ty(i,o.shape,o.dtype,r);return e.makeTensorInfo(a,l,c)}const _D={kernelName:ni,backendName:"cpu",kernelFunc:Os};function ue(n,t,e,s){return e==null?({inputs:o,backend:r})=>{const{a:i,b:a}=o,l=r;at([i,a],n);const c=l.data.get(i.dataId).values,u=l.data.get(a.dataId).values,h=i.dtype==="string"?rs(c):c,d=i.dtype==="string"?rs(u):u,p=s||i.dtype,[f,m]=t(i.shape,a.shape,h,d,p);return l.makeTensorInfo(m,p,f)}:({inputs:o,backend:r})=>{const{a:i,b:a}=o,l=r;if(i.dtype==="complex64"||a.dtype==="complex64"){const c=Os({inputs:{x:i},backend:l,attrs:{dtype:"complex64"}}),u=l.data.get(c.dataId),h=u.complexTensorInfos.real,d=u.complexTensorInfos.imag,p=l.data.get(h.dataId).values,f=l.data.get(d.dataId).values,m=Os({inputs:{x:a},backend:l,attrs:{dtype:"complex64"}}),g=l.data.get(m.dataId),x=g.complexTensorInfos.real,b=g.complexTensorInfos.imag,w=l.data.get(x.dataId).values,y=l.data.get(b.dataId).values,[$,v,N]=e(i.shape,a.shape,p,f,w,y),T=l.makeTensorInfo(N,"float32",$),k=l.makeTensorInfo(N,"float32",v),S=qe({inputs:{real:T,imag:k},backend:l});return l.disposeIntermediateTensorInfo(c),l.disposeIntermediateTensorInfo(m),l.disposeIntermediateTensorInfo(T),l.disposeIntermediateTensorInfo(k),S}else{const c=l.data.get(i.dataId).values,u=l.data.get(a.dataId).values,h=s||i.dtype,[d,p]=t(i.shape,a.shape,c,u,h);return l.makeTensorInfo(p,h,d)}}}function Op(n){return(t,e,s,o,r,i)=>{const a=gt(t,e),l=U(a),c=a.length,u=ut(a),h=we("float32",l),d=we("float32",l),p=po(t,a),f=po(e,a),m=os(s,o),g=os(r,i),x=t.length,b=ut(t),w=e.length,y=ut(e);if(p.length+f.length===0)for(let $=0;$<h.length;$++){const v=$%m.length,N=$%g.length,T=n(m[v*2],m[v*2+1],g[N*2],g[N*2+1]);h[$]=T.real,d[$]=T.imag}else for(let $=0;$<h.length;$++){const v=ko($,c,u),N=v.slice(-x);p.forEach(D=>N[D]=0);const T=Sn(N,x,b),k=v.slice(-w);f.forEach(D=>k[D]=0);const S=Sn(k,w,y),C=n(m[T*2],m[T*2+1],g[S*2],g[S*2+1]);h[$]=C.real,d[$]=C.imag}return[h,d,a]}}const Ey=oe(((n,t)=>n+t)),OD=Op(((n,t,e,s)=>({real:n+e,imag:t+s}))),Yo=ue(tr,Ey,OD),LD={kernelName:tr,backendName:"cpu",kernelFunc:Yo};function Lp(n,t,e,s,o){const r=U(s),i=$e(o,e);for(let a=0;a<n.length;a++){const l=n[a];if(l<0)throw new Error("Input x must be non-negative!");l>=o||(r>0?i[l]+=t[a]:i[l]+=1)}return i}function Ry(n,t,e,s=!1){const o=n.shape[0],r=n.shape[1],i=Ct([o,e],t.dtype);for(let a=0;a<o;a++)for(let l=0;l<r;l++){const c=n.get(a,l);if(c<0)throw new Error("Input x must be non-negative!");c>=e||(s?i.set(1,a,c):t.size>0?i.set(i.get(a,c)+t.get(a,l),a,c):i.set(i.get(a,c)+1,a,c))}return i}const Ay=oe(((n,t)=>n&t)),MD=ue(ch,Ay),PD={kernelName:ch,backendName:"cpu",kernelFunc:MD};function Kn(n){return(t,e,s)=>{const o=Jt(e,t.length);for(let r=0;r<t.length;++r)o[r]=n(t[r],s);return o}}function Dt(n,t,e){const s=Kn(t);return Us(n,s,e)}function Us(n,t,e){return({inputs:s,attrs:o,backend:r})=>{const{x:i}=s;at(i,n);const a=r,l=a.data.get(i.dataId).values;let c;if(i.dtype==="string"){if(!Array.isArray(l))throw new Error("String tensor's value was not an instance of Array");c=rs(l)}else c=l;const u=e||i.dtype,h=t(c,u,o);return a.makeTensorInfo(i.shape,u,h)}}const Dy=Kn(n=>Math.ceil(n)),zD=Us(si,Dy),BD={kernelName:si,backendName:"cpu",kernelFunc:zD};function Fy(n,t,e,s){const o=Jt(e,U(t));if(s&&e!=="string"){let r=0;n.forEach(i=>{const a=U(i.shape);o.set(i.vals,r),r+=a})}else{let r=0;n.forEach(i=>{const a=e==="string"?rs(i.vals):i.vals;let l=0;for(let c=0;c<i.shape[0];++c){const u=c*t[1]+r;for(let h=0;h<i.shape[1];++h)o[u+h]=a[l++]}r+=i.shape[1]})}return o}const _y=oe((n,t)=>n===t?1:0),Oy=ue(Al,_y,null,"bool"),VD={kernelName:Al,backendName:"cpu",kernelFunc:Oy};const Ly=Kn(n=>Math.exp(n)),My=Us(ui,Ly,"float32"),WD={kernelName:ui,backendName:"cpu",kernelFunc:My};const Py=Kn(n=>Math.expm1(n)),UD=Us(hi,Py),GD={kernelName:hi,backendName:"cpu",kernelFunc:UD};const zy=Kn(n=>Math.floor(n)),HD=Us(di,zy),qD={kernelName:di,backendName:"cpu",kernelFunc:HD};const By=oe((n,t)=>Math.floor(n/t)),jD=ue(pi,By,null,"int32"),KD={kernelName:pi,backendName:"cpu",kernelFunc:jD};function Vy(n,t,e,s,o,r,i,a,l){const c=Ct([s,r],e);for(let u=0;u<s;u++){const h=[];let d=0;for(let p=0;p<o;p++){const f=n[u*o+p];d+=f*i[p],h.push(f)}if(d<0||d>=l/r)throw new Error(`Invalid indices: ${h} does not index into ${a}`);for(let p=0;p<r;p++)c.values[u*r+p]=t.get(...t.indexToLoc(d*r+p))}return c}function Wy(n,t,e){const s=Ct(e,n.dtype);for(let o=0;o<s.size;++o){const i=s.indexToLoc(o).slice(),a=i[0],l=i[2],c=t.locToIndex([a,l]);i[2]=t.values[c];const u=n.locToIndex(i);0<=u&&u<n.values.length&&(s.values[o]=n.values[u])}return s}const Uy=oe((n,t)=>n>t?1:0),XD=ue(Ol,Uy,null,"bool"),YD={kernelName:Ol,backendName:"cpu",kernelFunc:XD};const Gy=oe((n,t)=>n>=t?1:0),ZD=ue(fi,Gy,null,"bool"),JD={kernelName:fi,backendName:"cpu",kernelFunc:ZD};const Hy=oe((n,t)=>n<t?1:0),QD=ue(Ml,Hy,null,"bool"),tF={kernelName:Ml,backendName:"cpu",kernelFunc:QD};const qy=oe((n,t)=>n<=t?1:0),eF=ue(Pl,qy,null,"bool"),nF={kernelName:Pl,backendName:"cpu",kernelFunc:eF};function jy(n,t,e){const s=(t-n)/(e-1),o=$e(e,"float32");o[0]=n;for(let r=1;r<o.length;r++)o[r]=o[r-1]+s;return o}const Ky=Kn(n=>Math.log(n)),sF=Us(yi,Ky),oF={kernelName:yi,backendName:"cpu",kernelFunc:sF};function Xy(n,t,e,s){const o=we(s,U(e));for(let r=0;r<o.length;++r){const i=r*t;let a=n[i];for(let l=0;l<t;++l){const c=n[i+l];(Number.isNaN(c)||c>a)&&(a=c)}o[r]=a}return o}const Yy=oe(((n,t)=>Math.max(n,t))),rF=ue($i,Yy),iF={kernelName:$i,backendName:"cpu",kernelFunc:rF};const Zy=oe(((n,t)=>Math.min(n,t))),aF=ue(Ci,Zy),lF={kernelName:Ci,backendName:"cpu",kernelFunc:aF};const Mp=oe(((n,t)=>n*t)),cF=Op(((n,t,e,s)=>({real:n*e-t*s,imag:n*s+t*e}))),Gc=ue(vi,Mp,cF),uF={kernelName:vi,backendName:"cpu",kernelFunc:Gc};function Jy(n,t,e){const s=as(-1,e);return Mp([],t,s,n,e)}function hF(n){const{inputs:t,backend:e}=n,{x:s}=t;at(s,"neg");const o=e.data.get(s.dataId).values,[r,i]=Jy(o,s.shape,s.dtype);return e.makeTensorInfo(i,s.dtype,r)}const dF={kernelName:Xl,backendName:"cpu",kernelFunc:hF};const Qy=oe(((n,t)=>n!==t?1:0)),pF=ue(Yl,Qy,null,"bool"),fF={kernelName:Yl,backendName:"cpu",kernelFunc:pF};function Pp(n,t,e,s,o){const r=t.length,i=U(t),a=ut(t),l=ut(o),c=we(e,U(o));for(let u=0;u<i;++u){const h=ko(u,r,a),d=new Array(h.length);for(let f=0;f<d.length;f++)d[f]=h[s[f]];const p=Sn(d,r,l);c[p]=n[u]}return c}function Ue(n){const{inputs:t,attrs:e,backend:s}=n,{x:o}=t,{perm:r}=e;at(o,"transpose");const i=o.shape.length,a=new Array(i);for(let h=0;h<a.length;h++)a[h]=o.shape[r[h]];const l=s.data.get(o.dataId).values,c=Pp(l,o.shape,o.dtype,r,a);return{dataId:s.write(c,a,o.dtype),shape:a,dtype:o.dtype}}const mF={kernelName:zo,backendName:"cpu",kernelFunc:Ue};function t1(n,t,e,s){const[o,r]=me(n,s),i=Ke(t,"int32"),a=$e(U(o),i),l=U(r);for(let c=0;c<a.length;++c){const u=c*l;let h=1;for(let d=0;d<l;++d)h*=e[u+d];a[c]=h}return{outVals:a,outShape:o,outDtype:i}}function gF(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;at(o,"prod");const a=o.shape.length,l=$t(r,o.shape),c=Xt(l,a);let u=l,h=o;const d=[];c!=null&&(h=Ue({inputs:{x:o},backend:e,attrs:{perm:c}}),d.push(h),u=ne(u.length,a));const p=e.data.get(h.dataId).values,{outVals:f,outShape:m,outDtype:g}=t1(h.shape,h.dtype,p,u);let x=m;return i&&(x=ee(m,l)),d.forEach(b=>e.disposeIntermediateTensorInfo(b)),e.makeTensorInfo(x,g,f)}const xF={kernelName:nc,backendName:"cpu",kernelFunc:gF};function bF(n,t,e){n.forEach((s,o)=>{if(s<0||s>=e){const r=ko(o,t.length,ut(t)).join(",");throw new Error(`indices[${r}] = ${s} is not in [0, ${e})`)}})}function yF(n,t){for(let e=0;e<n.length;++e){const s=n[e],o=e===n.length-1?t:n[e+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>o)throw new Error("Ragged splits must not point past values");for(let r=1;r<s.length;++r)if(s[r-1]>s[r])throw new Error("Ragged splits must be sorted in ascending order")}}function wF(n,t,e,s){const o=[];let r=0;const i=t.length-1+e.length,a=new Array(i).fill(null).map(()=>[0]);yF(e,s);let l=1;for(let c=0;c<t.length-1;++c){l*=t[c];const u=t[c+1];for(let h=1;h<l+1;++h)a[c].push(h*u)}for(let c=0;c<n.length;++c){let u=n[c],h=n[c]+1;for(let d=0;d<e.length;++d){const p=e[d],f=d+t.length-1;if(f>=0){const m=a[f],g=m[m.length-1]-p[u];for(let x=u;x<h;++x)a[f].push(p[x+1]+g)}u=p[u],h=p[h]}h!==u&&(o.push([u,h]),r+=h-u)}return{outSplits:a,valueSlices:o,numValues:r}}function $F(n){const t=[];for(let e=0;e<n.length;++e){const s=n[e].length,o=Jt("int32",s);t.push(o),n[e].forEach((r,i)=>o[i]=r)}return t}function fm(n,t){const e=n.slice(0,t);for(;e.length<t;)e.push(1);for(let s=t;s<n.length;s++)e[t-1]*=n[s];return e}function CF(n,t,e,s,o,r){const i=fm(t,2)[1],a=fm(r,2)[1];let l=0;for(const c of e)for(let u=c[0];u<c[1];++u){for(let h=0;h<s;++h)o[l*a+h]=n[u*i+h];++l}}function IF(n,t,e,s,o){const r=t.slice();r[0]=o;const i=Jt(e,U(r)),a=n.length,l=a===0?0:a/t[0];return CF(n,t,s,l,i,r),[i,r]}function e1(n,t,e,s,o,r,i,a){if(n.length===0)throw new Error("paramsNestedSplits must be non empty");if(t[0].length===0)throw new Error("Split tensors must not be scalars");const l=t[0][0]-1;if(bF(r,i,l),s.length===0)throw new Error("params.rank must be nonzero");const c=s[0],{outSplits:u,valueSlices:h,numValues:d}=wF(r,i,n,c),p=$F(u),f=IF(e,s,o,h,d);return[p,f[0],f[1]]}const mm=2147483647;function n1(n,t,e,s,o,r,i){if(t.length>1)throw new Error("starts must be a scalar or vector");if(o.length>1)throw new Error("limits must be a scalar or vector");if(i.length>1)throw new Error("deltas must be a scalar or vector");const a=t.length===0,l=o.length===0,c=i.length===0,u=[];a||u.push(t[0]),l||u.push(o[0]),c||u.push(i[0]);for(let g=1;g<u.length;++g)if(u[g]!==u[g-1])throw new Error("starts, limits, and deltas must have the same shape");const h=u.length===0?1:u[0],d=Jt("int32",h+1);d[0]=0;for(let g=0;g<h;++g){const x=a?n[0]:n[g],b=l?s[0]:s[g],w=c?r[0]:r[g];if(w===0)throw new Error("Requires delta != 0");let y;if(w>0&&b<x||w<0&&b>x)y=0;else if(y=Math.ceil(Math.abs((b-x)/w)),y>mm)throw new Error(`Requires ((limit - start) / delta) <= ${mm}`);d[g+1]=d[g]+y}const p=d[h],f=Jt(e,p);let m=0;for(let g=0;g<h;++g){const x=d[g+1]-d[g];let b=a?n[0]:n[g];const w=c?r[0]:r[g];for(let y=0;y<x;++y)f[m++]=b,b+=w}return[d,f]}var cn=vn;class al{constructor(t,e,s,o,r,i,a,l,c,u){this.shape=t,this.shapeShape=e,this.values=s,this.valuesShape=o,this.valuesDType=r,this.defaultValue=i,this.defaultValueShape=a,this.rowPartitionValues=l,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=_x(u),this.raggedRank=Ox(this.rowPartitionTypes)}getRowPartitionTypeByDimension(t){return this.rowPartitionTypes[0]===cn.FIRST_DIM_SIZE?this.rowPartitionTypes[t+1]:this.rowPartitionTypes[t]}getRowPartitionTensor(t){return this.rowPartitionTypes[0]===cn.FIRST_DIM_SIZE?this.rowPartitionValues[t+1]:this.rowPartitionValues[t]}getMaxWidth(t){const e=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case cn.VALUE_ROWIDS:return al.getMaxWidthValueRowID(e);case cn.ROW_SPLITS:return al.getMaxWidthRowSplit(e);default:throw new Error(`Cannot handle partition type ${cn[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(t){const e=t.length;if(e===0||e===1)return 0;let s=0;for(let o=0;o<e-1;++o){const r=t[o+1]-t[o];r>s&&(s=r)}return s}static getMaxWidthValueRowID(t){const e=t.length;if(e===0)return 0;let s=0,o=t[0],r=0;for(let i=1;i<e;++i){const a=t[i];a!==o&&(o=a,r=Math.max(i-s,r),s=i)}return Math.max(e-s,r)}tensorShapeFromTensor(t,e,s=!0){if(e.length===0){if(t[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return xm(t,s)}calculateOutputSize(t){const e=this.valuesShape,s=this.defaultValueShape;Lx(s,e);const o=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=Fx(this.raggedRank,o,e);i[0]<0&&(i[0]=t);for(let a=1;a<=this.raggedRank;++a)i[a]<0&&(i[a]=this.getMaxWidth(a));return i}calculateFirstParentOutputIndex(t,e,s){const o=Math.min(t,s),r=[];let i=0;for(let a=0;a<o;++a,i+=e)r.push(i);for(let a=o;a<t;++a)r.push(-1);return I(r.length===t,()=>"Final length of result must be equal to firstDimension."),r}calculateOutputIndexRowSplit(t,e,s,o){const r=t.length,i=[];for(let a=0;a<r-1;++a){const l=t[a+1]-t[a];let c=Math.min(o,l),u=e[a];u===-1&&(c=0);for(let h=0;h<c;++h)i.push(u),u+=s;for(let h=0;h<l-c;++h)i.push(-1)}if(r>0&&i.length!==t[r-1])throw new Error("Invalid row split size.");return i}calculateOutputIndexValueRowID(t,e,s,o){const r=t.length,i=[];if(r===0)return[];let a=0,l=t[0];if(l>=e.length)throw new Error(`Got currentValueRowId=${l}, which is not less than ${e.length}`);let c=e[l];i.push(c);for(let u=1;u<r;++u){const h=t[u];if(h===l)c>=0&&(++a,a<o?c+=s:c=-1);else{if(a=0,l=h,h>=e.length)throw new Error(`Got nextValueRowId=${h} which is not less than ${e.length}`);c=e[h]}i.push(c)}if(i.length!==t.length)throw new Error("Invalid row ids.");return i}calculateOutputIndex(t,e,s,o){const r=this.getRowPartitionTensor(t),i=this.getRowPartitionTypeByDimension(t);switch(i){case cn.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(r,e,s,o);case cn.ROW_SPLITS:if(r.length-1>e.length)throw new Error(`Row partition size is greater than output size: ${r.length-1} > ${e.length}`);return this.calculateOutputIndexRowSplit(r,e,s,o);default:throw new Error(`Unsupported partition type: ${cn[i]}`)}}getFirstDimensionSize(){const t=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const e=this.rowPartitionTypes[0];switch(e){case cn.FIRST_DIM_SIZE:return t[0];case cn.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case cn.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${cn[e]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const e=this.getFirstDimensionSize(),s=this.calculateOutputSize(e),o=new Array(this.raggedRank+1);o[o.length-1]=1;for(let l=o.length-2;l>=0;--l)o[l]=o[l+1]*s[l+1];const r=xm(s,!1),i=Jt(this.valuesDType,U(r));if(o[0]*s[0]>0){let l=this.calculateFirstParentOutputIndex(e,o[0],s[0]);for(let c=1;c<=this.raggedRank;++c)l=this.calculateOutputIndex(c-1,l,o[c],s[c]);this.setOutput(this.raggedRank,l,i,r)}return[r,i]}setOutput(t,e,s,o){if(s.length===0)return;const r=this.values,i=s;let a=o.slice();a=a.slice(t+1);const l=U(a),c=e.length;let u=this.defaultValue;if(u.length!==l&&u.length!==1){const f=this.defaultValueShape;z(()=>{const m=L(u,f);u=Er(m,a).dataSync()})}let h=0,d=0,p=0;for(let f=0;f<=c;++f){let m=f<c?e[f]:-1;if(m===p){++p;continue}if(d<p){const g=r.subarray(h*l),x=i.subarray(d*l),b=(p-d)*l;gm(x,g,b)}if(f>=c){const g=s.length;m=Math.floor(g/l)}if(m>p)if(this.defaultValue.length===1)i.subarray(p*l,m*l).fill(this.defaultValue[0]),p=m;else for(;m>p;){const g=i.slice(p*l);gm(g,u,l),++p}m<0?(h=f+1,d=p):(h=f,d=p,p=d+1)}}}function gm(n,t,e){for(let s=0;s<e;s++)n[s]=t[s]}function xm(n,t){const e=[];for(let s of n){if(s<0){if(!t)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}e.push(s)}return e}function s1(n,t,e,s,o,r,i,a,l,c){return new al(n,t,e,s,o,r,i,a,l,c).compute()}function o1(n,t,e,s){const o=n===t,r=n<t&&e<0,i=t<n&&e>1;if(o||r||i)return $e(0,s);const a=Math.abs(Math.ceil((t-n)/e)),l=$e(a,s);t<n&&e===1&&(e=-1),l[0]=n;for(let c=1;c<l.length;c++)l[c]=l[c-1]+e;return l}const r1=Kn(n=>1/Math.sqrt(n)),vF=Us(Ri,r1),kF={kernelName:Ri,backendName:"cpu",kernelFunc:vF};function Qs(n,t,e,s,o,r,i,a,l,c){const u=[s/o,o],h=n.values,d=t.values;if(s===0)return Ct(e,t.dtype);const p=l instanceof ye?l:Ct(u,t.dtype);typeof l=="string"||typeof l=="number"?p.values.fill(l):typeof l=="boolean"&&p.values.fill(+l);for(let f=0;f<r;f++){const m=[];let g=0;for(let x=0;x<i;x++){const b=h[f*i+x];m.push(b),g+=b*a[x]}if(g<0||g>=s/o)throw new Error(`Invalid indices: ${m} does not index into ${e}`);for(let x=0;x<o;x++)c?p.values[g*o+x]+=d[f*o+x]:p.values[g*o+x]=t.rank===0?d[0]:d[f*o+x]}return p}const SF=Kn(n=>1/(1+Math.exp(-n))),i1=Dt(Oi,n=>1/(1+Math.exp(-n))),NF={kernelName:Oi,backendName:"cpu",kernelFunc:i1};function a1(n,t,e,s,o){const r=Ud(s,t,e),i=U(e),a=ut(s);if(r){const h=Gd(t,a);return o==="string"?n.slice(h,h+i):n.subarray(h,h+i)}const l=o==="string"?rs(n):n,c=Ct(s,o,l),u=Ct(e,o);for(let h=0;h<u.size;++h){const d=u.indexToLoc(h),p=d.map((f,m)=>f+t[m]);u.set(c.get(...p),...d)}return o==="string"?nb(u.values):u.values}function wo(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,size:i}=s;at(o,"slice");const[a,l]=Nc(o,r,i);Vd(o,a,l);const c=e.data.get(o.dataId).values,u=a1(c,a,l,o.shape,o.dtype);return e.makeTensorInfo(l,o.dtype,u)}const TF={kernelName:lc,backendName:"cpu",kernelFunc:wo};function l1(n,t,e,s,o,r,i){const a=t[0],l=r[0],c=new Array(l),u=new Array(a),h=t[1];if(l===0){if(a!==0)throw new Error(Ux(a));const g=Jt(e,0),x=Jt(o,0);return[g,[0,h],x,c,u]}let d=!0,p=0;const f=new Array(l).fill(0);for(let g=0;g<a;++g){const x=n[g*h];if(x<0)throw new Error(Gx(g,x));if(x>=l)throw new Error(Hx(g,x,l));++f[x],d=d&&x>=p,p=x}let m=!0;for(let g=0;g<l;++g){const x=f[g]===0;c[g]=x,m=m&&!x,f[g]=Math.max(f[g],1),g>0&&(f[g]+=f[g-1])}if(m&&d){const g=n,x=s;for(let b=0;b<a;++b)u[b]=b;return[g,[a,h],x,c,u]}else{const g=f[l-1],x=Jt(e,g*h),b=Jt(o,g),w=new Array(l).fill(0);for(let y=0;y<a;++y){const $=n[y*h],v=w[$],N=($===0?0:f[$-1])+v;w[$]++;for(let T=0;T<h;++T)x[N*h+T]=n[y*h+T];b[N]=s[y],u[y]=N}for(let y=0;y<l;++y)if(w[y]===0){const v=y===0?0:f[y-1];x[v*h+0]=y;for(let N=1;N<h;++N)x[v*h+N]=0;b[v]=i}return[x,[g,h],b,c,u]}}function c1(n,t,e,s,o){const r=U(s),i=t[0],a=o.length,l=[];let c=1,u=-1;for(let g=0;g<a;++g){const x=o[g];if(x===-1){if(u!==-1)throw new Error(qx(u,g));u=g,l.push(1)}else{if(x<0)throw new Error(jx(g,x));c*=x,l.push(x)}}if(u!==-1){if(c<=0)throw new Error(Kx());const g=Math.trunc(r/c);if(c*g!==r)throw new Error(Xx(s,l));l[u]=g}if(U(l)!==r)throw new Error(Yx(s,l));const d=s.length,p=[];if(d>0){p[d-1]=1;for(let g=d-2;g>=0;--g)p[g]=p[g+1]*s[g+1]}const f=[];if(a>0){f[a-1]=1;for(let g=a-2;g>=0;--g)f[g]=f[g+1]*l[g+1]}const m=Jt(e,i*a);for(let g=0;g<i;++g){let x=0;for(let b=0;b<d;++b)x+=n[g*d+b]*p[b];for(let b=0;b<a;++b)m[g*a+b]=Math.trunc(x/f[b]),x%=f[b]}return[m,[i,a],l]}function zp(n,t,e,s,o,r=!1,i=0){const a=s.length,l=[t[0],n.length/t[0]],c=l[1],h=a>0?o[a-1]+1:0;if(h<0)throw new Error(Au());const d=t.slice();d[0]=h;const p=d.reduce((w,y)=>w*y,1),f=Jt(e,p);if(a===0)return h>0&&f.fill(i),[f,d];if(h<=0)throw new Error(Au());let m=0,g=1,x=0,b=o[m];for(;;){let w=0;if(g<a){if(w=o[g],b===w){++g;continue}if(b>=w)throw new Error(Zx())}if(b<0||b>=h)throw new Error(Jx(b,h));b>x&&f.fill(i,x*c,b*c);for(let y=m;y<g;++y){const $=s[y];if($<0||$>=l[0])throw new Error(Qx(y,s[y],l[0]));for(let v=0;v<c;v++)f[b*c+v]+=n[$*c+v]}if(r)for(let y=0;y<c;y++)f[b*c+y]/=g-m;if(m=g,++g,x=b+1,b=w,g>a)break}return x<h&&f.fill(i,x*c,h*c),[f,d]}const EF=Kn(n=>Math.sqrt(n)),RF=Dt(Mi,n=>Math.sqrt(n)),AF={kernelName:Mi,backendName:"cpu",kernelFunc:RF};const u1=oe(((n,t)=>{const e=n-t;return e*e})),DF=ue(Pi,u1),FF={kernelName:Pi,backendName:"cpu",kernelFunc:DF};const h1=Kn((n,t)=>{const{pattern:e,replaceGlobal:s,rewrite:o}=t;return n.replace(new RegExp(e,s?"g":""),o)}),_F=Us(pc,h1),OF={kernelName:pc,backendName:"cpu",kernelFunc:_F};function d1(n,t,e,s){const o=Ct(n,t.dtype);for(let r=0;r<o.size;r++){const i=o.indexToLoc(r),a=new Array(i.length);for(let l=0;l<a.length;l++)a[l]=i[l]*e[l]+s[l];o.set(t.get(...a),...i)}return o}class LF{constructor(t,e,s,o,r,i){this.separator=es(t),this.nGramWidths=e,this.leftPad=es(s),this.rightPad=es(o),this.padWidth=r,this.preserveShort=i}getPadWidth(t){return Math.min(this.padWidth<0?t-1:this.padWidth,t-1)}getNumNGrams(t,e){const s=this.getPadWidth(e);return Math.max(0,t+2*s-e+1)}createNGrams(t,e,s,o,r,i){for(let a=0;a<r;++a){const l=this.getPadWidth(i),c=Math.max(0,l-a),u=Math.max(0,l-(r-(a+1))),h=i-(c+u),d=e+(c>0?0:a-l);let p=0;p+=c*this.leftPad.length;for(let b=0;b<h;++b)p+=t[d+b].length;p+=u*this.rightPad.length;const f=c+u+h-1;p+=f*this.separator.length,s[o+a]=new Uint8Array(p);const m=s[o+a];let g=0;const x=b=>b.forEach(w=>m[g++]=w);for(let b=0;b<c;++b)x(this.leftPad),x(this.separator);for(let b=0;b<h-1;++b)x(t[d+b]),x(this.separator);if(h>0){x(t[d+h-1]);for(let b=0;b<u;++b)x(this.separator),x(this.rightPad)}else{for(let b=0;b<u-1;++b)x(this.rightPad),x(this.separator);x(this.rightPad)}}}compute(t,e){const s=t.length,o=e.length;if(o>0){let l=e[0];if(l!==0)throw new Error(`First split value must be 0, got ${l}`);for(let c=1;c<o;++c){let u=e[c]>=l;if(u=u&&e[c]<=s,!u)throw new Error(`Invalid split value ${e[c]}, must be in [${l}, ${s}]`);l=e[c]}if(l!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${l}`)}const r=o-1,i=Jt("int32",o);if(s===0||o===0){const l=new Array(s);for(let c=0;c<=r;++c)i[c]=0;return[l,i]}i[0]=0;for(let l=1;l<=r;++l){const c=e[l]-e[l-1];let u=0;this.nGramWidths.forEach(h=>{u+=this.getNumNGrams(c,h)}),this.preserveShort&&c>0&&u===0&&(u=1),i[l]=i[l-1]+u}const a=new Array(i[r]);for(let l=0;l<r;++l){const c=e[l];let u=i[l];if(this.nGramWidths.forEach(h=>{const d=e[l+1]-e[l],p=this.getNumNGrams(d,h);this.createNGrams(t,c,a,u,p,h),u+=p}),this.preserveShort&&u===i[l]){const h=e[l+1]-e[l];if(h===0)continue;const d=h+2*this.padWidth;this.createNGrams(t,c,a,u,1,d)}}return[a,i]}}function p1(n,t,e,s,o,r,i,a){return new LF(e,s,o,r,i,a).compute(n,t)}function MF(n,t,e,s){if(!n.length)return;if(t.length===0){for(let r=0;r<n.length;++r)s.push(n.subarray(r,r+1));return}if(t.length===1){const r=t[0];let i=n.indexOf(r);for(;i!==-1;){const a=n.subarray(0,i);(!e||a.length!==0)&&s.push(a),n=n.subarray(i+1),i=n.indexOf(r)}(!e||n.length!==0)&&s.push(n);return}let o=0;for(let r=0;r<n.length+1;r++)if(r===n.length||t.indexOf(n[r])!==-1){const i=n.subarray(o,r);(!e||i.length!==0)&&s.push(i),o=r+1}}function f1(n,t,e){const s=n.length,o=[];let r=0,i=0;const a=new Array(s);for(let d=0;d<s;++d){const p=o.length;MF(n[d],t,e,o);const f=o.length-p;a[d]=f,r+=f,i=Math.max(i,f)}const l=Jt("int32",r*2),c=new Array(r),u=[s,i];let h=0;for(let d=0;d<s;++d)for(let p=0;p<a[d];++p)l[h*2]=d,l[h*2+1]=p,c[h]=o[h],++h;return[l,c,u]}function m1(n,t){const e=Jt("int32",n.length);for(let s=0;s<n.length;++s)e[s]=bg(n[s]).modulo(t).getLowBitsUnsigned();return e}const g1=oe(((n,t)=>n-t)),PF=Op(((n,t,e,s)=>({real:n-e,imag:t-s}))),Bp=ue(zi,g1,PF),zF={kernelName:zi,backendName:"cpu",kernelFunc:Bp};function x1(n,t){const e=new Array(n.rank);for(let o=0;o<e.length;o++)e[o]=n.shape[o]*t[o];const s=Ct(e,n.dtype);for(let o=0;o<s.values.length;++o){const r=s.indexToLoc(o),i=new Array(n.rank);for(let l=0;l<i.length;l++)i[l]=r[l]%n.shape[l];const a=n.locToIndex(i);s.values[o]=n.values[a]}return s}const Ir=(n,t)=>{const e=t.value-n.value;return e===0?n.index-t.index:e};function b1(n,t,e=0,s=n.length-1){for(;s>e;){if(s-e>600){const a=s-e+1,l=t-e+1,c=Math.log(a),u=.5*Math.exp(2*c/3),h=.5*Math.sqrt(c*u*(a-u)/a)*Math.sign(l-a/2),d=Math.max(e,Math.floor(t-l*u/a+h)),p=Math.min(s,Math.floor(t+(a-l)*u/a+h));b1(n,t,d,p)}const o=n[t];let r=e,i=s;for(Jn(n,e,t),Ir(n[s],o)>0&&Jn(n,e,s);r<i;){for(Jn(n,r,i),r++,i--;Ir(n[r],o)<0;)r=r+1;for(;Ir(n[i],o)>0;)i=i-1}Ir(n[e],o)===0?Jn(n,e,i):(i=i+1,Jn(n,i,s)),i<=t&&(e=i+1),t<=i&&(s=i-1)}}function y1(n,t,e,s,o){const r=t[t.length-1],[i,a]=[n.length/r,r],l=we(e,i*s),c=we("int32",i*s);for(let h=0;h<i;h++){const d=h*a,p=n.subarray(d,d+a);let f=new Array(p.length);p.forEach((b,w)=>f[w]={value:b,index:w}),s<f.length&&(b1(f,s),f=f.slice(0,s)),o&&f.sort(Ir);const m=h*s,g=l.subarray(m,m+s),x=c.subarray(m,m+s);for(let b=0;b<s;b++)g[b]=f[b].value,x[b]=f[b].index}const u=t.slice();return u[u.length-1]=s,[Ct(u,e,l),Ct(u,"int32",c)]}function w1(n,t,e,s){const o=$t(t,e)[0],r=[1,e[0],1];for(let f=0;f<o;f++)r[0]*=e[f];r[1]=e[o];for(let f=o+1;f<e.length;f++)r[2]*=e[f];const i=new Map,a=new Int32Array(e[o]),l=new ye(r,s,n),c=[],u=r[0]===1&&r[2]===1;for(let f=0;f<e[o];f++){let m;if(u)m=n[f].toString();else{const x=[];for(let b=0;b<r[0];b++)for(let w=0;w<r[2];w++)x.push(l.get(b,f,w));m=x.join(",")}const g=i.get(m);if(g!=null)a[f]=g;else{const x=i.size;i.set(m,x),a[f]=x,c.push(f)}}const h=r.slice();h[1]=i.size;const d=new ye(h,s);c.forEach((f,m)=>{for(let g=0;g<r[0];g++)for(let x=0;x<r[2];x++)d.set(l.get(g,f,x),g,m,x)});const p=e.slice();return p[o]=h[1],{outputValues:d.values,outputShape:p,indices:a}}const BF=Object.freeze(Object.defineProperty({__proto__:null,addImpl:Ey,bincountImpl:Lp,bincountReduceImpl:Ry,bitwiseAndImpl:Ay,castImpl:Ty,ceilImpl:Dy,concatImpl:Fy,equalImpl:_y,expImpl:Ly,expm1Impl:Py,floorDivImpl:By,floorImpl:zy,gatherNdImpl:Vy,gatherV2Impl:Wy,greaterEqualImpl:Gy,greaterImpl:Uy,lessEqualImpl:qy,lessImpl:Hy,linSpaceImpl:jy,logImpl:Ky,maxImpl:Xy,maximumImpl:Yy,minimumImpl:Zy,multiplyImpl:Mp,negImpl:Jy,notEqualImpl:Qy,prodImpl:t1,raggedGatherImpl:e1,raggedRangeImpl:n1,raggedTensorToTensorImpl:s1,rangeImpl:o1,rsqrtImpl:r1,scatterImpl:Qs,sigmoidImpl:SF,simpleAbsImpl:Ny,sliceImpl:a1,sparseFillEmptyRowsImpl:l1,sparseReshapeImpl:c1,sparseSegmentReductionImpl:zp,sqrtImpl:EF,squaredDifferenceImpl:u1,staticRegexReplaceImpl:h1,stridedSliceImpl:d1,stringNGramsImpl:p1,stringSplitImpl:f1,stringToHashBucketFastImpl:m1,subImpl:g1,tileImpl:x1,topKImpl:y1,transposeImpl:Pp,uniqueImpl:w1},Symbol.toStringTag,{value:"Module"}));kg("cpu",()=>new Uc,1);const $1=Dt(li,n=>n>=0?n:Math.exp(n)-1),VF={kernelName:li,backendName:"cpu",kernelFunc:$1};function C1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{alpha:r}=s;at([o],"leakyRelu");const i=U(o.shape),a=e.data.get(o.dataId).values,l=we("float32",i);for(let c=0;c<a.length;c++)l[c]=a[c]<0?r*a[c]:a[c];return e.makeTensorInfo(o.shape,"float32",l)}const WF={kernelName:Ll,backendName:"cpu",kernelFunc:C1};const UF=oe((n,t)=>n<0?t*n:n);function I1(n){const{inputs:t,backend:e}=n,{x:s,alpha:o}=t;at([s,o],"prelu");const r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,[a,l]=UF(s.shape,o.shape,r,i,"float32");return e.makeTensorInfo(l,"float32",a)}const GF={kernelName:ec,backendName:"cpu",kernelFunc:I1};const v1=Dt(Ni,n=>Math.max(0,n)),HF={kernelName:Ni,backendName:"cpu",kernelFunc:v1};const k1=Dt(Ti,n=>Math.min(Math.max(0,n),6)),qF={kernelName:Ti,backendName:"cpu",kernelFunc:k1};function ll(n,t,e,s,o){if(e==="linear")return jn({inputs:{x:t},backend:n});if(e==="relu")return v1({inputs:{x:t},backend:n});if(e==="elu")return $1({inputs:{x:t},backend:n});if(e==="relu6")return k1({inputs:{x:t},backend:n});if(e==="prelu")return I1({inputs:{x:t,alpha:s},backend:n});if(e==="leakyrelu")return C1({inputs:{x:t},backend:n,attrs:{alpha:o}});if(e==="sigmoid")return i1({inputs:{x:t},backend:n});throw new Error(`Activation ${e} has not been implemented for the CPU backend.`)}function Wt(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{shape:r}=s,i=U(o.shape),a=Qu(r,i),l=U(a);I(i===l,()=>`The new shape (${a}) has ${l} elements and the old shape (${o.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`),e.incRef(o.dataId);const c=e.data.get(o.dataId);if(c.complexTensorInfos!=null){const u=c.complexTensorInfos.real,h=c.complexTensorInfos.imag;u.shape=a,h.shape=a}return{dataId:o.dataId,shape:a,dtype:o.dtype}}const jF={kernelName:sc,backendName:"cpu",kernelFunc:Wt};function S1(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r}=t,{transposeA:i,transposeB:a}=s;at([o,r],"matMul");const l=o.shape.length,c=r.shape.length,u=i?o.shape[l-2]:o.shape[l-1],h=a?r.shape[c-1]:r.shape[c-2],d=i?o.shape[l-1]:o.shape[l-2],p=a?r.shape[c-2]:r.shape[c-1],f=o.shape.slice(0,-2),m=r.shape.slice(0,-2),g=U(f),x=U(m),w=gt(o.shape.slice(0,-2),r.shape.slice(0,-2)).concat([d,p]);I(u===h,()=>`Error in matMul: inner shapes (${u}) and (${h}) of Tensors with shapes ${o.shape} and ${r.shape} and transposeA=${i} and transposeB=${a} must match.`);const y=i?[g,u,d]:[g,d,u],$=a?[x,p,h]:[x,h,p],v=Wt({inputs:{x:o},backend:e,attrs:{shape:y}}),N=Wt({inputs:{x:r},backend:e,attrs:{shape:$}}),T=i?v.shape[1]:v.shape[2],k=i?v.shape[2]:v.shape[1],S=a?N.shape[1]:N.shape[2],C=Math.max(g,x),D=e.data.get(v.dataId).values,O=e.data.get(N.dataId).values,P=ut(v.shape),M=ut(N.shape),[V,H,W]=i?[P[0],1,P[1]]:[P[0],P[1],1],[q,j,K]=a?[1,M[1],M[0]]:[M[1],1,M[0]],Y=k*S,Z=Ct([C,k,S],v.dtype),tt=Z.values,Q=e.blockSize;for(let ot=0;ot<C;ot++){const ht=ot%g,pt=ot%x;for(let dt=0;dt<k;dt+=Q){const bt=Math.min(dt+Q,k);for(let wt=0;wt<S;wt+=Q){const Rt=Math.min(wt+Q,S);for(let Pt=0;Pt<T;Pt+=Q){const Yt=Math.min(Pt+Q,T);for(let zt=dt;zt<bt;zt++)for(let Ft=wt;Ft<Rt;Ft++){let jt=0;for(let Ut=Pt;Ut<Yt;Ut++){const Xn=D[ht*V+zt*H+Ut*W],ge=O[Ut*q+Ft*j+pt*K];jt+=Xn*ge}tt[ot*Y+(zt*S+Ft)]+=jt}}}}}return e.disposeIntermediateTensorInfo(v),e.disposeIntermediateTensorInfo(N),e.makeTensorInfo(w,Z.dtype,Z.values)}const KF={kernelName:$l,backendName:"cpu",kernelFunc:S1};function XF(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r,bias:i,preluActivationWeights:a}=t,{transposeA:l,transposeB:c,activation:u,leakyreluAlpha:h}=s;let d,p,f;const m=[];d=S1({inputs:{a:o,b:r},attrs:{transposeA:l,transposeB:c},backend:e}),i&&(p=Yo({inputs:{a:d,b:i},backend:e}),m.push(d),d=p),u&&(f=ll(e,d,u,a,h),m.push(d),d=f);for(const x of m)e.disposeIntermediateTensorInfo(x);return d}const YF={kernelName:Wa,backendName:"cpu",kernelFunc:XF};const ZF=Dt(Xr,n=>Math.acos(n)),JF={kernelName:Xr,backendName:"cpu",kernelFunc:ZF};const QF=Dt(Yr,n=>Math.acosh(n)),t_={kernelName:Yr,backendName:"cpu",kernelFunc:QF};function e_(n){const{inputs:t,backend:e}=n,s=t;at(t,"addN");const o=s.map(a=>e.data.get(a.dataId).values),r=Ct(s[0].shape,s[0].dtype),i=r.values;for(let a=0;a<s.length;a++){const l=o[a];for(let c=0;c<i.length;c++)i[c]+=l[c]}return e.makeTensorInfo(r.shape,r.dtype,r.values)}const n_={kernelName:sh,backendName:"cpu",kernelFunc:e_};function s_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;at(o,"all");const a=$t(r,o.shape);let l=a;const c=Xt(l,o.shape.length);let u=o;c!=null&&(u=Ue({inputs:{x:o},backend:e,attrs:{perm:c}}),l=ne(l.length,o.shape.length)),Ie("all",l,u.shape.length);const[h,d]=me(u.shape,l),p=U(d),f=$e(U(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const $=m[b+y];w=w&&$}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ee(h,a),b=Wt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const o_={kernelName:oh,backendName:"cpu",kernelFunc:s_};function r_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;at(o,"any");const a=$t(r,o.shape);let l=a;const c=Xt(l,o.shape.length);let u=o;c!=null&&(u=Ue({inputs:{x:o},backend:e,attrs:{perm:c}}),l=ne(l.length,o.shape.length)),Ie("any",l,u.shape.length);const[h,d]=me(u.shape,l),p=U(d),f=$e(U(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const $=m[b+y];w=w||$}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ee(h,a),b=Wt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const i_={kernelName:rh,backendName:"cpu",kernelFunc:r_};function a_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;at(o,"argMax");let i=$t(r,o.shape);const a=Xt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=Ue({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=ne(i.length,l.shape.length)),i=[i[0]],Ie("argMax",i,l.shape.length);const[u,h]=me(l.shape,i),d=U(u),p=$e(d,"int32"),f=U(h),m=e.data.get(l.dataId).values;for(let g=0;g<p.length;++g){const x=g*f;let b=m[x],w=0;for(let y=0;y<f;++y){const $=m[x+y];$>b&&(b=$,w=y)}p[g]=w}return c.forEach(g=>e.disposeIntermediateTensorInfo(g)),e.makeTensorInfo(u,"int32",p)}const l_={kernelName:xl,backendName:"cpu",kernelFunc:a_};function c_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;at(o,"argMin");let i=$t(r,o.shape);const a=Xt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=Ue({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=ne(i.length,l.shape.length)),i=[i[0]],Ie("argMin",i,l.shape.length);const[u,h]=me(l.shape,i),d=U(u),p=$e(d,"int32"),f=U(h),m=e.data.get(l.dataId).values;for(let g=0;g<p.length;++g){const x=g*f;let b=m[x],w=0;for(let y=0;y<f;++y){const $=m[x+y];$<b&&(b=$,w=y)}p[g]=w}return c.forEach(g=>e.disposeIntermediateTensorInfo(g)),e.makeTensorInfo(u,"int32",p)}const u_={kernelName:bl,backendName:"cpu",kernelFunc:c_};const h_=Dt(Zr,n=>Math.asin(n)),d_={kernelName:Zr,backendName:"cpu",kernelFunc:h_};const p_=Dt(Jr,n=>Math.asinh(n)),f_={kernelName:Jr,backendName:"cpu",kernelFunc:p_};const m_=Dt(Qr,n=>Math.atan(n)),g_={kernelName:Qr,backendName:"cpu",kernelFunc:m_};const x_=oe((n,t)=>Math.atan2(n,t)),b_=ue(ei,x_),y_={kernelName:ei,backendName:"cpu",kernelFunc:b_};const w_=Dt(ti,n=>Math.atanh(n)),$_={kernelName:ti,backendName:"cpu",kernelFunc:w_};function Vp(n,t,e,s,o,r){const i=o.strideHeight,a=o.strideWidth,l=o.dilationHeight,c=o.dilationWidth,u=o.effectiveFilterHeight,h=o.effectiveFilterWidth,d=o.padInfo.top,p=o.padInfo.left,f=r==="max"?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,m=Ct(o.outShape,e),g=m.values,x=o.outShape[1]*o.outShape[2]*o.outShape[3],b=o.outShape[2]*o.outShape[3],w=o.outShape[3];for(let y=0;y<o.batchSize;++y){const $=y*x,v=y*s[0];for(let N=0;N<o.inChannels;++N)for(let T=0;T<o.outHeight;++T){const k=T*i-d,S=Math.max(0,k),C=Math.min(o.inHeight,u+k),D=$+T*b;for(let O=0;O<o.outWidth;++O){const P=O*a-p,M=Math.max(0,P),V=Math.min(o.inWidth,h+P);let H=f,W=0,q=0;for(let K=S;K<C;K+=l){const Y=v+K*s[1];for(let Z=M;Z<V;Z+=c){const tt=Y+Z*s[2],Q=n[tt+N];r==="max"&&Q>H?H=Q:r==="avg"&&(W+=Q,q++)}if(isNaN(H))break}const j=D+O*w+N;g[j]=r==="avg"?W/q:H}}}return m}function N1(n,t,e,s,o=!1,r=!1){const i=Ct(s.outShape,"int32"),a=s.strideHeight,l=s.strideWidth,c=s.dilationHeight,u=s.dilationWidth,h=s.effectiveFilterHeight,d=s.effectiveFilterWidth,p=s.padInfo.top,f=s.padInfo.left,m=Ct(t,e,n);for(let g=0;g<s.batchSize;++g)for(let x=0;x<s.inChannels;++x)for(let b=0;b<s.outHeight;++b){const w=b*a-p;let y=w;for(;y<0;)y+=c;const $=Math.min(s.inHeight,h+w);for(let v=0;v<s.outWidth;++v){const N=v*l-f;let T=N;for(;T<0;)T+=u;const k=Math.min(s.inWidth,d+N);let S=Number.NEGATIVE_INFINITY,C=-1;for(let D=y;D<$;D+=c){const O=D-w;for(let P=T;P<k;P+=u){const M=P-N,V=m.get(g,D,P,x);V>S&&(S=V,o?C=r?((g*s.inHeight+D)*s.inWidth+P)*s.inChannels+x:(D*s.inWidth+P)*s.inChannels+x:C=O*d+M)}}i.set(C,g,b,v,x)}}return i}function T1(n,t,e,s,o,r){const i=o.strideDepth,a=o.strideHeight,l=o.strideWidth,c=o.dilationDepth,u=o.dilationHeight,h=o.dilationWidth,d=o.effectiveFilterDepth,p=o.effectiveFilterHeight,f=o.effectiveFilterWidth,m=o.padInfo.front,g=o.padInfo.top,x=o.padInfo.left,b=r==="max"?Number.NEGATIVE_INFINITY:Number.POSITIVE_INFINITY,w=Ct(o.outShape,e),y=w.values,$=o.outShape[1]*o.outShape[2]*o.outShape[3]*o.outShape[4],v=o.outShape[2]*o.outShape[3]*o.outShape[4],N=o.outShape[3]*o.outShape[4],T=o.outShape[4];for(let k=0;k<o.batchSize;++k){const S=k*$,C=k*s[0];for(let D=0;D<o.inChannels;++D)for(let O=0;O<o.outDepth;++O){const P=O*i-m;let M=P;for(;M<0;)M+=c;const V=Math.min(o.inDepth,d+P),H=S+O*v;for(let W=0;W<o.outHeight;++W){const q=W*a-g;let j=q;for(;j<0;)j+=u;const K=Math.min(o.inHeight,p+q),Y=H+W*N;for(let Z=0;Z<o.outWidth;++Z){const tt=Z*l-x;let Q=tt;for(;Q<0;)Q+=h;const ot=Math.min(o.inWidth,f+tt),ht=Y+Z*T;let pt=b,dt=0,bt=0;for(let Rt=M;Rt<V;Rt+=c){const Pt=C+Rt*s[1];for(let Yt=j;Yt<K;Yt+=u){const zt=Pt+Yt*s[2];for(let Ft=Q;Ft<ot;Ft+=h){const jt=zt+Ft*s[3],Ut=n[jt+D];if(r==="max"&&Ut>pt?pt=Ut:r==="avg"&&(dt+=Ut,bt++),isNaN(pt))break}if(isNaN(pt))break}if(isNaN(pt))break}const wt=ht+D;y[wt]=r==="avg"?dt/Math.max(bt,1):pt}}}}return w}function C_(n,t){const e=Ct(t.outShape,"int32"),s=t.strideDepth,o=t.strideHeight,r=t.strideWidth,i=t.dilationDepth,a=t.dilationHeight,l=t.dilationWidth,c=t.effectiveFilterDepth,u=t.effectiveFilterHeight,h=t.effectiveFilterWidth,d=t.padInfo.front,p=t.padInfo.top,f=t.padInfo.left;for(let m=0;m<t.batchSize;++m)for(let g=0;g<t.inChannels;++g)for(let x=0;x<t.outDepth;++x){const b=x*s-d;let w=b;for(;w<0;)w+=i;const y=Math.min(t.inDepth,c+b);for(let $=0;$<t.outHeight;++$){const v=$*o-p;let N=v;for(;N<0;)N+=a;const T=Math.min(t.inHeight,u+v);for(let k=0;k<t.outWidth;++k){const S=k*r-f;let C=S;for(;C<0;)C+=l;const D=Math.min(t.inWidth,h+S);let O=Number.NEGATIVE_INFINITY,P=-1;for(let M=w;M<y;M+=i){const V=M-b;for(let H=N;H<T;H+=a){const W=H-v;for(let q=C;q<D;q+=l){const j=q-S,K=n.get(m,M,H,q,g);K>=O&&(O=K,P=V*u*h+W*u+j)}}}e.set(P,m,x,$,k,g)}}}return e}function I_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;at(o,"avgPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;I(Te(i,c),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=xn(o.shape,r,i,c,a,l);let h;if(u.filterWidth===1&&u.filterHeight===1&&Tt(u.inShape,u.outShape))h=jn({inputs:{x:o},backend:e});else{const d=e.data.get(o.dataId).values,p=ut(o.shape),f=Vp(d,o.shape,o.dtype,p,u,"avg");h=e.makeTensorInfo(u.outShape,o.dtype,f.values)}return h}const v_={kernelName:yl,backendName:"cpu",kernelFunc:I_};function k_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s;at(o,"avgPool3d");const u=cs(o.shape,r,i,1,a,l,c),h=e.data.get(o.dataId).values,d=T1(h,o.shape,o.dtype,ut(o.shape),u,"avg");return e.makeTensorInfo(d.shape,"float32",d.values)}const S_={kernelName:wl,backendName:"cpu",kernelFunc:k_};function N_(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,{filterSize:i,strides:a,pad:l,dimRoundingMode:c}=s;at([o,r],"avgPool3DGrad");const u=cs(r.shape,i,a,1,l,c),h=u.strideDepth,d=u.strideHeight,p=u.strideWidth,f=u.filterDepth,m=u.filterHeight,g=u.filterWidth,x=u.dilationDepth,b=u.dilationHeight,w=u.dilationWidth,y=u.effectiveFilterDepth,$=u.effectiveFilterHeight,v=u.effectiveFilterWidth,N=y-1-u.padInfo.front,T=v-1-u.padInfo.left,k=$-1-u.padInfo.top,S=Ct(r.shape,"float32"),C=1/(f*m*g),D=e.bufferSync(o);for(let O=0;O<u.batchSize;++O)for(let P=0;P<u.inChannels;++P)for(let M=0;M<u.inDepth;++M)for(let V=0;V<u.inHeight;++V)for(let H=0;H<u.inWidth;++H){const W=M-N,q=V-k,j=H-T;let K=0;for(let Y=0;Y<y;Y+=x){const Z=(W+Y)/h;if(!(Z<0||Z>=u.outDepth||Math.floor(Z)!==Z))for(let tt=0;tt<$;tt+=b){const Q=(q+tt)/d;if(!(Q<0||Q>=u.outHeight||Math.floor(Q)!==Q))for(let ot=0;ot<v;ot+=w){const ht=(j+ot)/p;if(ht<0||ht>=u.outWidth||Math.floor(ht)!==ht)continue;const pt=D.get(O,Z,Q,ht,P);K+=pt}}}S.set(K*C,O,M,V,H,P)}return e.makeTensorInfo(S.shape,S.dtype,S.values)}const T_={kernelName:ah,backendName:"cpu",kernelFunc:N_};function E_(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r;at([o,r],"avgPoolGrad");const{filterSize:a,strides:l,pad:c}=s,u=xn(i.shape,a,l,1,c),h=u.strideHeight,d=u.strideWidth,p=u.filterHeight,f=u.filterWidth,m=u.dilationHeight,g=u.dilationWidth,x=u.effectiveFilterHeight,b=u.effectiveFilterWidth,w=b-1-u.padInfo.left,y=x-1-u.padInfo.top,$=Ct(i.shape,"float32"),v=1/(p*f),N=e.data.get(o.dataId).values,T=Ct(o.shape,"float32",N);for(let k=0;k<u.batchSize;++k)for(let S=0;S<u.inChannels;++S)for(let C=0;C<u.inHeight;++C)for(let D=0;D<u.inWidth;++D){const O=C-y,P=D-w;let M=0;for(let V=0;V<x;V+=m){const H=(O+V)/h;if(!(H<0||H>=u.outHeight||Math.floor(H)!==H))for(let W=0;W<b;W+=g){const q=(P+W)/d;if(q<0||q>=u.outWidth||Math.floor(q)!==q)continue;const j=T.get(k,H,q,S);M+=j}}$.set(M*v,k,C,D,S)}return e.makeTensorInfo($.shape,$.dtype,$.values)}const R_={kernelName:ih,backendName:"cpu",kernelFunc:E_};function A_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,scale:r,offset:i,mean:a,variance:l}=t;I(a.shape.length===l.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),I(i==null||a.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),I(r==null||a.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks."),at([o,a,l,r,i],"batchNorm");let{varianceEpsilon:c}=s;c==null&&(c=.001);const u=e.data.get(o.dataId).values,h=e.data.get(a.dataId).values,d=e.data.get(l.dataId).values,p=r?e.data.get(r.dataId).values:new Float32Array([1]),f=i?e.data.get(i.dataId).values:new Float32Array([0]),m=new Float32Array(u.length),g=f.length,x=p.length,b=d.length,w=h.length;let y=0,$=0,v=0,N=0;for(let T=0;T<u.length;++T)m[T]=f[y++]+(u[T]-h[$++])*p[v++]/Math.sqrt(d[N++]+c),y>=g&&(y=0),$>=w&&($=0),v>=x&&(v=0),N>=b&&(N=0);return e.makeTensorInfo(o.shape,o.dtype,m)}const D_={kernelName:Fl,backendName:"cpu",kernelFunc:A_};function F_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,crops:i}=s;at([o],"batchToSpaceND");const a=r.reduce((x,b)=>x*b),l=Zi(o.shape,r,a),c=Ji(l.length,r.length),u=Qi(o.shape,r,a),h=Xd(i,r.length),d=Yd(u,i,r.length),p=Wt({inputs:{x:o},backend:e,attrs:{shape:l}}),f=Ue({inputs:{x:p},backend:e,attrs:{perm:c}}),m=Wt({inputs:{x:f},backend:e,attrs:{shape:u}}),g=wo({inputs:{x:m},backend:e,attrs:{begin:h,size:d}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),g}const __={kernelName:Cl,backendName:"cpu",kernelFunc:F_};function O_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i}=s,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,c=Lp(a,l,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,c)}const L_={kernelName:lh,backendName:"cpu",kernelFunc:O_};function M_(n){const{inputs:t,backend:e}=n,{s0:s,s1:o}=t,r=e.data.get(s.dataId).values,i=e.data.get(o.dataId).values,a=gt(Array.from(r),Array.from(i));return e.makeTensorInfo([a.length],"int32",Int32Array.from(a))}const P_={kernelName:tg,backendName:"cpu",kernelFunc:M_};const z_=Dt(oi,(n,t)=>{const e=t;return n>e.clipValueMax?e.clipValueMax:n<e.clipValueMin?e.clipValueMin:n}),B_={kernelName:oi,backendName:"cpu",kernelFunc:z_};const V_=n=>{const{x:t}=n.inputs,e=n.backend,s=new Float32Array(U(t.shape)),o=e.data.get(t.dataId),r=o.complexTensorInfos.real,i=o.complexTensorInfos.imag,a=e.data.get(r.dataId).values,l=e.data.get(i.dataId).values;for(let c=0;c<a.length;c++){const u=a[c],h=l[c];s[c]=Math.hypot(u,h)}return e.makeOutput(s,t.shape,"float32")},W_={kernelName:Il,backendName:"cpu",kernelFunc:V_};function Zo(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.data.get(s.dataId).complexTensorInfos.imag,r=e.data.get(o.dataId).values;return e.makeTensorInfo(o.shape,o.dtype,r)}const U_={kernelName:Nh,backendName:"cpu",kernelFunc:Zo};function Jo(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s,r=$t(o,t[0].shape)[0],i=t.map(m=>m.shape);qd(i,r);let a=Vn(t.map(m=>m.shape),r);if(U(a)===0)return e.makeTensorInfo(a,t[0].dtype,[]);const l=t.filter(m=>U(m.shape)>0);if(l.length===1)return jn({inputs:{x:l[0]},backend:e});if(l[0].dtype==="complex64"){const m=l.map(y=>yo({inputs:{input:y},backend:e})),g=l.map(y=>Zo({inputs:{input:y},backend:e})),x=Jo({inputs:m,backend:e,attrs:{axis:r}}),b=Jo({inputs:g,backend:e,attrs:{axis:r}}),w=qe({inputs:{real:x,imag:b},backend:e});return m.forEach(y=>e.disposeIntermediateTensorInfo(y)),g.forEach(y=>e.disposeIntermediateTensorInfo(y)),e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(b),w}const c=l.map(m=>{const x=[-1,U(m.shape.slice(r))];return Wt({inputs:{x:m},backend:e,attrs:{shape:x}})}),u=c.map(m=>({vals:e.data.get(m.dataId).values,shape:m.shape}));a=Vn(c.map(m=>m.shape),1);const h=c[0].shape[0]===1,d=Fy(u,a,t[0].dtype,h),p=Vn(l.map(m=>m.shape),r),f=e.makeTensorInfo(p,t[0].dtype,d);return c.forEach(m=>e.disposeIntermediateTensorInfo(m)),f}const G_={kernelName:vl,backendName:"cpu",kernelFunc:Jo};function E1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dataFormat:l,dilations:c,dimRoundingMode:u}=s;at([o,r],"conv2d");const h=us(l),d=Ce(o.shape,r.shape,i,c,a,u,!1,h),p=d.filterHeight,f=d.filterWidth,m=d.dilationHeight,g=d.dilationWidth,x=d.padInfo.left,b=d.padInfo.top,w=d.dataFormat==="channelsLast",y=new ye(d.outShape,o.dtype),$=ut(o.shape),v=ut(r.shape),N=$[0],T=w?$[1]:$[2],k=w?$[2]:1,S=w?1:$[1],C=y.strides[0],D=w?y.strides[1]:y.strides[2],O=w?y.strides[2]:1,P=w?1:y.strides[1],M=e.data.get(o.dataId).values,V=e.data.get(r.dataId).values,H=y.values;for(let W=0;W<d.batchSize;++W){const q=W*N,j=W*C;for(let K=0;K<d.outHeight;++K){const Y=j+K*D,Z=K*d.strideHeight-b;for(let tt=0;tt<p;++tt){const Q=Z+tt*m;if(Q<0||Q>=d.inHeight)continue;const ot=tt*v[0],ht=q+Q*T;for(let pt=0;pt<d.outWidth;++pt){const dt=Y+pt*O,bt=pt*d.strideWidth-x;for(let wt=0;wt<f;++wt){const Rt=bt+wt*g;if(Rt<0||Rt>=d.inWidth)continue;const Pt=ot+wt*v[1],Yt=ht+Rt*k;let zt=Pt;for(let Ft=0;Ft<d.inChannels;++Ft){const jt=M[Yt+Ft*S];for(let Ut=0;Ut<d.outChannels;++Ut)H[dt+Ut*P]+=jt*V[zt+Ut];zt+=d.outChannels}}}}}}return e.makeTensorInfo(y.shape,y.dtype,H)}const H_={kernelName:kl,backendName:"cpu",kernelFunc:E1};function q_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,dataFormat:l,dimRoundingMode:c,filterShape:u}=s;at([o,r],"conv2dBackpropFilter");const h=us(l),d=Ce(o.shape,u,i,1,a,c,!1,h),{strideHeight:p,strideWidth:f,filterHeight:m,filterWidth:g}=d,x=d.dataFormat==="channelsLast",b=new ye(d.filterShape,"float32"),w=d.padInfo.left,y=d.padInfo.top,$=e.data.get(o.dataId).values,v=e.data.get(r.dataId).values,N=new ye(o.shape,o.dtype,$),T=new ye(r.shape,r.dtype,v);for(let k=0;k<m;++k){const S=Math.max(0,Math.ceil((y-k)/p)),C=Math.min(d.outHeight,(d.inHeight+y-k)/p);for(let D=0;D<g;++D){const O=Math.max(0,Math.ceil((w-D)/f)),P=Math.min(d.outWidth,(d.inWidth+w-D)/f);for(let M=0;M<d.inChannels;++M)for(let V=0;V<d.outChannels;++V){let H=0;for(let W=0;W<d.batchSize;++W)for(let q=S;q<C;++q){const j=k+q*p-y;for(let K=O;K<P;++K){const Y=D+K*f-w;x?H+=N.get(W,j,Y,M)*T.get(W,q,K,V):H+=N.get(W,M,j,Y)*T.get(W,V,q,K)}}b.set(H,k,D,M,V)}}}return e.makeTensorInfo(b.shape,b.dtype,b.values)}const j_={kernelName:hh,backendName:"cpu",kernelFunc:q_};function K_(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{inputShape:i,strides:a,pad:l,dataFormat:c,dimRoundingMode:u}=s;at([o,r],"conv2dBackpropInput");const h=ut(r.shape),d=ut(o.shape);let p=us(c);const f=Ce(i,r.shape,a,1,l,u,!1,p),m=new ye(f.inShape,"float32"),g=m.values,x=e.data.get(o.dataId).values,b=e.data.get(r.dataId).values,[w,y,$]=h,{batchSize:v,filterHeight:N,filterWidth:T,inChannels:k,inHeight:S,inWidth:C,outChannels:D,outHeight:O,outWidth:P,strideHeight:M,strideWidth:V}=f;p=f.dataFormat;const H=N-1-f.padInfo.top,W=T-1-f.padInfo.left,q=p==="channelsLast",j=m.strides[0],K=q?m.strides[1]:m.strides[2],Y=q?m.strides[2]:1,Z=q?1:m.strides[1],tt=d[0],Q=q?d[1]:d[2],ot=q?d[2]:1,ht=q?1:d[1];for(let pt=0;pt<v;++pt)for(let dt=0;dt<k;++dt)for(let bt=0;bt<S;++bt){const wt=bt-H,Rt=Math.max(0,Math.ceil(wt/M)),Pt=Math.min(O,(N+wt)/M);for(let Yt=0;Yt<C;++Yt){const zt=Yt-W,Ft=Math.max(0,Math.ceil(zt/V)),jt=Math.min(P,(T+zt)/V);let Ut=0;for(let ge=Rt;ge<Pt;++ge){const ms=ge*M-wt;for(let Je=Ft;Je<jt;++Je){const Hs=Je*V-zt,wn=tt*pt+Q*ge+ot*Je,Yn=w*(N-1-ms)+y*(T-1-Hs)+$*dt;for(let gs=0;gs<D;++gs){const xs=x[wn+ht*gs],bs=b[Yn+gs];Ut+=xs*bs}}}const Xn=j*pt+K*bt+Y*Yt+Z*dt;g[Xn]=Ut}}return e.makeTensorInfo(m.shape,m.dtype,m.values)}const X_={kernelName:Sl,backendName:"cpu",kernelFunc:K_};function Y_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s;at([o,r],"conv3d");const c=Ms(o.shape,r.shape,i,l,a),{filterDepth:u,filterHeight:h,filterWidth:d,dilationDepth:p,dilationHeight:f,dilationWidth:m,padInfo:g}=c,x=g.front,b=g.left,w=g.top,y=new ye(c.outShape,o.dtype),$=e.data.get(o.dataId).values,v=e.data.get(r.dataId).values,N=y.values,T=ut(o.shape),k=ut(r.shape);for(let S=0;S<c.batchSize;++S){const C=S*T[0],D=S*y.strides[0];for(let O=0;O<c.outDepth;++O){const P=D+O*y.strides[1],M=O*c.strideDepth-x;for(let V=0;V<u;++V){const H=M+V*p;if(H<0||H>=c.inDepth)continue;const W=V*k[0],q=C+H*T[1];for(let j=0;j<c.outHeight;++j){const K=P+j*y.strides[2],Y=j*c.strideHeight-w;for(let Z=0;Z<h;++Z){const tt=Y+Z*f;if(tt<0||tt>=c.inHeight)continue;const Q=W+Z*k[1],ot=q+tt*T[2];for(let ht=0;ht<c.outWidth;++ht){const pt=K+ht*c.outChannels,dt=ht*c.strideWidth-b;for(let bt=0;bt<d;++bt){const wt=dt+bt*m;if(wt<0||wt>=c.inWidth)continue;const Rt=Q+bt*k[2],Pt=ot+wt*c.inChannels;let Yt=Rt;for(let zt=0;zt<c.inChannels;++zt){const Ft=$[Pt+zt];for(let jt=0;jt<c.outChannels;++jt)N[pt+jt]+=Ft*v[Yt+jt];Yt+=c.outChannels}}}}}}}}return e.makeTensorInfo(y.shape,y.dtype,y.values)}const Z_={kernelName:Nl,backendName:"cpu",kernelFunc:Y_};function J_(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,filterShape:l}=s;at([o,r],"conv3dBackpropFilterV2");const c=ut(o.shape),u=ut(r.shape),h=Ms(o.shape,l,i,1,a),d=h.strideDepth,p=h.strideHeight,f=h.strideWidth,m=h.filterDepth,g=h.filterHeight,x=h.filterWidth,b=new ye(h.filterShape,"float32"),w=b.values,[y,$,v,N]=b.strides,T=e.data.get(r.dataId).values,[k,S,C,D]=u,O=e.data.get(o.dataId).values,[P,M,V,H]=c,W=h.padInfo.front,q=h.padInfo.left,j=h.padInfo.top;for(let K=0;K<m;++K){const Y=Math.max(0,Math.ceil((W-K)/d)),Z=Math.min(h.outDepth,(h.inDepth+W-K)/d),tt=K*y;for(let Q=0;Q<g;++Q){const ot=Math.max(0,Math.ceil((j-Q)/p)),ht=Math.min(h.outHeight,(h.inHeight+j-Q)/p),pt=Q*$+tt;for(let dt=0;dt<x;++dt){const bt=Math.max(0,Math.ceil((q-dt)/f)),wt=Math.min(h.outWidth,(h.inWidth+q-dt)/f),Rt=dt*v+pt;for(let Pt=0;Pt<h.inChannels;++Pt){const Yt=Pt*N+Rt;for(let zt=0;zt<h.outChannels;++zt){let Ft=0;for(let jt=0;jt<h.batchSize;++jt){const Ut=jt*P,Xn=jt*k;for(let ge=Y;ge<Z;++ge){const Je=(K+ge*d-W)*M+Ut,Hs=ge*S+Xn;for(let wn=ot;wn<ht;++wn){const gs=(Q+wn*p-j)*V+Je,xs=wn*C+Hs;for(let bs=bt;bs<wt;++bs){const Yc=(dt+bs*f-q)*H+gs,Zc=bs*D+xs;Ft+=O[Yc+Pt]*T[Zc+zt]}}}}w[Yt+zt]=Ft}}}}}return e.makeTensorInfo(b.shape,b.dtype,b.values)}const Q_={kernelName:dh,backendName:"cpu",kernelFunc:J_};function tO(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{pad:i,strides:a,inputShape:l}=s;at([o],"conv3dBackpropInputV2");const c=ut(o.shape),u=ut(r.shape),h=Ms(l,r.shape,a,1,i),d=new ye(h.inShape,"float32"),p=d.values,[f,m,g,x]=d.strides,b=e.data.get(o.dataId).values,[w,y,$,v]=c,N=e.data.get(r.dataId).values,[T,k,S,C]=u,{batchSize:D,filterDepth:O,filterHeight:P,filterWidth:M,inChannels:V,inDepth:H,inHeight:W,inWidth:q,outChannels:j,outDepth:K,outHeight:Y,outWidth:Z,strideDepth:tt,strideHeight:Q,strideWidth:ot}=h,ht=O-1-h.padInfo.front,pt=P-1-h.padInfo.top,dt=M-1-h.padInfo.left;for(let bt=0;bt<D;++bt)for(let wt=0;wt<V;++wt)for(let Rt=0;Rt<H;++Rt){const Pt=Rt-ht,Yt=Math.max(0,Math.ceil(Pt/tt)),zt=Math.min(K,(O+Pt)/tt);for(let Ft=0;Ft<W;++Ft){const jt=Ft-pt,Ut=Math.max(0,Math.ceil(jt/Q)),Xn=Math.min(Y,(P+jt)/Q);for(let ge=0;ge<q;++ge){const ms=ge-dt,Je=Math.max(0,Math.ceil(ms/ot)),Hs=Math.min(Z,(M+ms)/ot);let wn=0;for(let Yn=Yt;Yn<zt;++Yn){const gs=Yn*tt-Pt;for(let xs=Ut;xs<Xn;++xs){const bs=xs*Q-jt;for(let mr=Je;mr<Hs;++mr){const Yc=mr*ot-ms,Zc=w*bt+y*Yn+$*xs+v*mr,l$=T*(O-1-gs)+k*(P-1-bs)+S*(M-1-Yc)+C*wt;for(let pa=0;pa<j;++pa){const c$=b[Zc+pa],u$=N[l$+pa];wn+=c$*u$}}}}p[f*bt+m*Rt+g*Ft+x*ge+wt]=wn}}}return e.makeTensorInfo(d.shape,d.dtype,d.values)}const eO={kernelName:ph,backendName:"cpu",kernelFunc:tO};const nO=Dt(ri,n=>Math.cos(n)),sO={kernelName:ri,backendName:"cpu",kernelFunc:nO};const oO=Dt(ii,n=>Math.cosh(n)),rO={kernelName:ii,backendName:"cpu",kernelFunc:oO};function iO(n){const{inputs:t,backend:e,attrs:s}=n,{image:o,boxes:r,boxInd:i}=t,{cropSize:a,method:l,extrapolationValue:c}=s,[u,h,d,p]=o.shape,f=r.shape[0],[m,g]=a,x=Ct([f,m,g,p],"float32"),b=e.data.get(r.dataId).values,w=e.data.get(i.dataId).values,y=e.data.get(o.dataId).values,$=ut(o.shape),v=ut(x.shape);for(let N=0;N<f;N++){const T=N*4,k=b[T],S=b[T+1],C=b[T+2],D=b[T+3],O=w[N];if(O>=u)continue;const P=m>1?(C-k)*(h-1)/(m-1):0,M=g>1?(D-S)*(d-1)/(g-1):0;for(let V=0;V<m;V++){const H=m>1?k*(h-1)+V*P:.5*(k+C)*(h-1);if(H<0||H>h-1){for(let W=0;W<g;W++)for(let q=0;q<p;q++){const j=q+W*v[2]+V*v[1]+N*v[0];x.values[j]=c}continue}if(l==="bilinear"){const W=Math.floor(H),q=Math.ceil(H),j=H-W;for(let K=0;K<g;K++){const Y=g>1?S*(d-1)+K*M:.5*(S+D)*(d-1);if(Y<0||Y>d-1){for(let ot=0;ot<p;ot++){const ht=ot+K*v[2]+V*v[1]+N*v[0];x.values[ht]=c}continue}const Z=Math.floor(Y),tt=Math.ceil(Y),Q=Y-Z;for(let ot=0;ot<p;ot++){let ht=ot+Z*$[2]+W*$[1]+O*$[0];const pt=y[ht];ht=ot+tt*$[2]+W*$[1]+O*$[0];const dt=y[ht];ht=ot+Z*$[2]+q*$[1]+O*$[0];const bt=y[ht];ht=ot+tt*$[2]+q*$[1]+O*$[0];const wt=y[ht],Rt=pt+(dt-pt)*Q,Pt=bt+(wt-bt)*Q;ht=ot+K*v[2]+V*v[1]+N*v[0],x.values[ht]=Rt+(Pt-Rt)*j}}}else for(let W=0;W<g;++W){const q=g>1?S*(d-1)+W*M:.5*(S+D)*(d-1);if(q<0||q>d-1){for(let Y=0;Y<p;Y++){const Z=Y+W*v[2]+V*v[1]+N*v[0];x.values[Z]=c}continue}const j=Math.round(q),K=Math.round(H);for(let Y=0;Y<p;Y++){const Z=Y+j*$[2]+K*$[1]+O*$[0],tt=Y+W*v[2]+V*v[1]+N*v[0];x.values[tt]=y[Z]}}}}return e.makeTensorInfo(x.shape,x.dtype,x.values)}const aO={kernelName:mh,backendName:"cpu",kernelFunc:iO};function lO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;at(o,"cumprod");const l=Xt([r],o.shape.length);let c=o;l!=null&&(c=Ue({inputs:{x:o},backend:e,attrs:{perm:l}}));const u=ne(1,o.shape.length)[0];if(u!==c.shape.length-1)throw new Error(`backend.cumprod in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${u}`);const h=Ke(c.dtype,"int32"),d=ml(U(c.shape),h),p=e.data.get(c.dataId).values,f=c.shape[c.shape.length-1],m=a?(x,b)=>x+f-b-1:(x,b)=>x+b;for(let x=0;x<p.length;x+=f)for(let b=0;b<f;b++){const w=m(x,b);if(b===0)d[w]=i?1:p[w];else{const y=m(x,b-1);d[w]=i?p[y]*d[y]:p[w]*d[y]}}const g=e.makeTensorInfo(c.shape,h,d);if(l!=null){const x=Ps(l),b=Ue({inputs:{x:g},backend:e,attrs:{perm:x}});return e.disposeIntermediateTensorInfo(g),e.disposeIntermediateTensorInfo(c),b}return g}const cO={kernelName:fh,backendName:"cpu",kernelFunc:lO};function uO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;at(o,"cumsum");const l=Xt([r],o.shape.length);let c=o;l!=null&&(c=Ue({inputs:{x:o},backend:e,attrs:{perm:l}}));const u=ne(1,o.shape.length)[0];if(u!==c.shape.length-1)throw new Error(`backend.cumsum in CPU expects an inner-most axis=${c.shape.length-1} but got axis=${u}`);const h=Ke(c.dtype,"int32"),d=$e(U(c.shape),h),p=e.data.get(c.dataId).values,f=c.shape[c.shape.length-1],m=a?(x,b)=>x+f-b-1:(x,b)=>x+b;for(let x=0;x<p.length;x+=f)for(let b=0;b<f;b++){const w=m(x,b);if(b===0)d[w]=i?0:p[w];else{const y=m(x,b-1);d[w]=i?p[y]+d[y]:p[w]+d[y]}}const g=e.makeTensorInfo(c.shape,h,d);if(l!=null){const x=Ps(l),b=Ue({inputs:{x:g},backend:e,attrs:{perm:x}});return e.disposeIntermediateTensorInfo(g),e.disposeIntermediateTensorInfo(c),b}return g}const hO={kernelName:Tl,backendName:"cpu",kernelFunc:uO};function dO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i,binaryOutput:a}=s;if(o.shape.length===1){const l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=Lp(l,c,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,u)}else if(o.shape.length===2){const l=e.bufferSync(o),c=e.bufferSync(r),u=Ry(l,c,i,a);return e.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const pO={kernelName:gh,backendName:"cpu",kernelFunc:dO};function fO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockSize:r,dataFormat:i}=s;I(i==="NHWC",()=>`Only NHWC dataFormat supported on CPU for depthToSpace. Got ${i}`);const a=o.shape[0],l=o.shape[1],c=o.shape[2],u=o.shape[3],h=l*r,d=c*r,p=u/(r*r),f=e.data.get(o.dataId).values,m=new Float32Array(a*h*d*p);let g=0;for(let x=0;x<a;++x)for(let b=0;b<h;++b){const w=Math.floor(b/r),y=b%r;for(let $=0;$<d;++$){const v=Math.floor($/r),N=$%r,T=(y*r+N)*p;for(let k=0;k<p;++k){const C=k+T+u*(v+c*(w+l*x));m[g++]=f[C]}}}return e.makeTensorInfo([a,h,d,p],o.dtype,m)}const mO={kernelName:xh,backendName:"cpu",kernelFunc:fO};function R1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l,dimRoundingMode:c}=s;at([o,r],"depthwiseConv2DNative");const u=ut(o.shape),h=ut(r.shape);let d=l;d==null&&(d=[1,1]),I(Te(i,d),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${d}'`);const p=Ce(o.shape,r.shape,i,d,a,c,!0),{filterHeight:f,filterWidth:m,dilationHeight:g,dilationWidth:x,padInfo:b}=p,w=b.left,y=b.top,$=p.outChannels/p.inChannels,v=new ye(p.outShape,o.dtype),N=e.data.get(o.dataId).values,T=e.data.get(r.dataId).values,k=v.values;for(let S=0;S<p.batchSize;++S){const C=S*u[0],D=S*v.strides[0];for(let O=0;O<p.outHeight;++O){const P=D+O*v.strides[1],M=O*p.strideHeight-y;for(let V=0;V<f;++V){const H=M+V*g;if(H<0||H>=p.inHeight)continue;const W=V*h[0],q=C+H*u[1];for(let j=0;j<p.outWidth;++j){const K=P+j*v.strides[2],Y=j*p.strideWidth-w;for(let Z=0;Z<m;++Z){const tt=Y+Z*x;if(tt<0||tt>=p.inWidth)continue;const Q=W+Z*h[1],ot=q+tt*p.inChannels;let ht=K,pt=Q;for(let dt=0;dt<p.inChannels;++dt){const bt=N[ot+dt];for(let wt=0;wt<$;++wt)k[ht+wt]+=bt*T[pt+wt];ht+=$,pt+=$}}}}}}return e.makeTensorInfo(v.shape,v.dtype,v.values)}const gO={kernelName:El,backendName:"cpu",kernelFunc:R1};function xO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,filterShape:u}=s;at([o,r],"depthwiseConv2dNativeBackpropFilter");const h=Ce(o.shape,u,i,a,l,c,!0),{strideHeight:d,strideWidth:p,filterHeight:f,filterWidth:m}=h,g=new ye(h.filterShape,"float32"),x=h.padInfo.left,b=h.padInfo.top,w=h.outChannels/h.inChannels,y=e.data.get(o.dataId).values,$=new ye(o.shape,o.dtype,y),v=e.data.get(r.dataId).values,N=new ye(r.shape,r.dtype,v);for(let T=0;T<f;++T){const k=Math.max(0,Math.ceil((b-T)/d)),S=Math.min(h.outHeight,(h.inHeight+b-T)/d);for(let C=0;C<m;++C){const D=Math.max(0,Math.ceil((x-C)/p)),O=Math.min(h.outWidth,(h.inWidth+x-C)/p);for(let P=0;P<h.outChannels;++P){const M=Math.trunc(P/w),V=P%w;let H=0;for(let W=0;W<h.batchSize;++W)for(let q=k;q<S;++q){const j=T+q*d-b;for(let K=D;K<O;++K){const Y=C+K*p-x;H+=$.get(W,j,Y,M)*N.get(W,q,K,P)}}g.set(H,T,C,M,V)}}}return e.makeTensorInfo(g.shape,g.dtype,g.values)}const bO={kernelName:bh,backendName:"cpu",kernelFunc:xO};function yO(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,inputShape:u}=s;at([o,r],"depthwiseConv2DNativeBackpropInput");const h=ut(o.shape),d=ut(r.shape),p=Ce(u,r.shape,i,a,l,c,!0),f=new ye(p.inShape,"float32"),m=f.values,[g,x,b]=f.strides,w=e.data.get(o.dataId).values,[y,$,v]=h,N=e.data.get(r.dataId).values,[T,k,S]=d,{batchSize:C,filterHeight:D,filterWidth:O,inChannels:P,inHeight:M,inWidth:V,outChannels:H,outHeight:W,outWidth:q,strideHeight:j,strideWidth:K}=p,Y=D-1-p.padInfo.top,Z=O-1-p.padInfo.left,tt=H/P;for(let Q=0;Q<C;++Q)for(let ot=0;ot<P;++ot)for(let ht=0;ht<M;++ht){const pt=ht-Y,dt=Math.max(0,Math.ceil(pt/j)),bt=Math.min(W,(D+pt)/j);for(let wt=0;wt<V;++wt){const Rt=wt-Z,Pt=Math.max(0,Math.ceil(Rt/K)),Yt=Math.min(q,(O+Rt)/K);let zt=0;for(let Ft=dt;Ft<bt;++Ft){const jt=Ft*j-pt;for(let Ut=Pt;Ut<Yt;++Ut){const Xn=Ut*K-Rt,ge=y*Q+$*Ft+v*Ut,ms=T*(D-1-jt)+k*(O-1-Xn)+S*ot;for(let Je=0;Je<tt;++Je){const Hs=ot*tt+Je,wn=w[ge+Hs],Yn=N[ms+Je];zt+=wn*Yn}}}m[g*Q+x*ht+b*wt+ot]=zt}}return e.makeTensorInfo(f.shape,f.dtype,f.values)}const wO={kernelName:yh,backendName:"cpu",kernelFunc:yO};function $O(n){const{inputs:t,backend:e}=n,{x:s}=t,o=U(s.shape),r=e.data.get(s.dataId).values,i=Ct([o,o],s.dtype),a=i.values;for(let c=0;c<r.length;c++)a[c*o+c]=r[c];const l=[...s.shape,...s.shape];return e.makeTensorInfo(l,i.dtype,i.values)}const CO={kernelName:eg,backendName:"cpu",kernelFunc:$O};const IO={kernelName:Rl,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o}=n,{strides:r,pad:i,dilations:a}=e,l=t,c=l.data.get(s.dataId).values,u=s.shape.length,h=l.data.get(o.dataId).values,d=o.shape.length,{batchSize:p,inHeight:f,inWidth:m,inChannels:g,outHeight:x,outWidth:b,padInfo:w,strideHeight:y,strideWidth:$,filterHeight:v,filterWidth:N,dilationHeight:T,dilationWidth:k,outShape:S}=Hi(s.shape,o.shape,r,i,"NHWC",a),C=U(S),D=S.length,O=Jt(s.dtype,C);for(let M=0;M<p;++M)for(let V=0;V<x;++V){const H=V*y-w.top;for(let W=0;W<b;++W){const q=W*$-w.left;for(let j=0;j<g;++j){let K=Number.MIN_SAFE_INTEGER;for(let Z=0;Z<v;++Z){const tt=H+Z*T;if(tt>=0&&tt<f)for(let Q=0;Q<N;++Q){const ot=q+Q*k;if(ot>=0&&ot<m){const ht=Sn([M,tt,ot,j],u,ut(s.shape)),pt=Sn([Z,Q,j],d,ut(o.shape)),dt=c[ht]+h[pt];dt>K&&(K=dt)}}}const Y=Sn([M,V,W,j],D,ut(S));O[Y]=K}}}return{dataId:l.write(Ls(O,s.dtype),S,s.dtype),shape:S,dtype:s.dtype}}};const vO={kernelName:mu,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o,dy:r}=n,{strides:i,pad:a,dilations:l}=e,c=t,u=pn(s.shape,c.data.get(s.dataId).values),h=pn(o.shape,c.data.get(o.dataId).values),{batchSize:d,inHeight:p,inWidth:f,inChannels:m,outHeight:g,outWidth:x,padInfo:b,strideHeight:w,strideWidth:y,filterHeight:$,filterWidth:v,dilationHeight:N,dilationWidth:T,outShape:k}=Hi(s.shape,o.shape,i,a,"NHWC",l);I(r.rank===k.length,()=>`Error in ${mu}, dy must have the same rank as output ${k.length}, but got ${r.rank}`);const S=pn(k,c.data.get(r.dataId).values),C=eh(o.shape,o.dtype);for(let O=0;O<d;++O)for(let P=0;P<g;++P){const M=P*w-b.top;for(let V=0;V<x;++V){const H=V*y-b.left;for(let W=0;W<m;++W){let q=Number.MIN_SAFE_INTEGER,j=0,K=0;for(let Y=0;Y<$;++Y){const Z=M+Y*N;if(Z>=0&&Z<p)for(let tt=0;tt<v;++tt){const Q=H+tt*T;if(Q>=0&&Q<f){const ot=u[O][Z][Q][W]+h[Y][tt][W];ot>q&&(q=ot,j=Y,K=tt)}}}C[j][K][W]+=S[O][P][V][W]}}}return{dataId:c.write(Ls(C,s.dtype),o.shape,o.dtype),shape:o.shape,dtype:o.dtype}}};const kO={kernelName:fu,backendName:"cpu",kernelFunc:({inputs:n,backend:t,attrs:e})=>{const{x:s,filter:o,dy:r}=n,{strides:i,pad:a,dilations:l}=e,c=t,u=pn(s.shape,c.data.get(s.dataId).values),h=pn(o.shape,c.data.get(o.dataId).values),{batchSize:d,inHeight:p,inWidth:f,inChannels:m,outHeight:g,outWidth:x,padInfo:b,strideHeight:w,strideWidth:y,filterHeight:$,filterWidth:v,dilationHeight:N,dilationWidth:T,outShape:k}=Hi(s.shape,o.shape,i,a,"NHWC",l);I(r.rank===k.length,()=>`Error in ${fu}, dy must have the same rank as output ${k.length}, but got ${r.rank}`);const S=pn(k,c.data.get(r.dataId).values),C=eh(s.shape,s.dtype);for(let O=0;O<d;++O)for(let P=0;P<g;++P){const M=P*w-b.top;for(let V=0;V<x;++V){const H=V*y-b.left;for(let W=0;W<m;++W){let q=Number.MIN_SAFE_INTEGER,j=M<0?0:M,K=H<0?0:H;for(let Y=0;Y<$;++Y){const Z=M+Y*N;if(Z>=0&&Z<p)for(let tt=0;tt<v;++tt){const Q=H+tt*T;if(Q>=0&&Q<f){const ot=u[O][Z][Q][W]+h[Y][tt][W];ot>q&&(q=ot,j=Z,K=Q)}}}C[O][j][K][W]+=S[O][P][V][W]}}}return{dataId:c.write(Ls(C,s.dtype),s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};function SO(n){const{inputs:t,backend:e,attrs:s}=n,{image:o}=t,{canvas:r,options:i}=s,{contextOptions:a,imageOptions:l}=i||{},c=l?.alpha||1,u=a?.contextType||"2d";if(u!=="2d")throw new Error(`Context type ${a.contextType} is not supported by the CPU backend.`);const h=r.getContext(u,a?.contextAttributes||{});if(h==null)throw new Error(`Could not get the context with ${u} type.`);const[d,p]=o.shape.slice(0,2),f=o.shape.length===2?1:o.shape[2],m=e.data.get(o.dataId).values,g=o.dtype==="float32"?255:1,x=new Uint8ClampedArray(p*d*4);for(let w=0;w<d*p;++w){const y=[0,0,0,255*c];for(let v=0;v<f;v++){const N=m[w*f+v];if(o.dtype==="float32"){if(N<0||N>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${N}.`)}else if(o.dtype==="int32"&&(N<0||N>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${N}.`);f===1?(y[0]=N*g,y[1]=N*g,y[2]=N*g):y[v]=N*g}const $=w*4;x[$+0]=Math.round(y[0]),x[$+1]=Math.round(y[1]),x[$+2]=Math.round(y[2]),x[$+3]=Math.round(y[3])}r.width=p,r.height=d;const b=new ImageData(x,p,d);return h.putImageData(b,0,0),o}const NO={kernelName:R$,backendName:"cpu",kernelFunc:SO};function la(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;at(o,"sum");let a;o.dtype==="bool"?a=Os({inputs:{x:o},backend:e,attrs:{dtype:"int32"}}):a=jn({inputs:{x:o},backend:e});const l=a.shape.length,c=$t(r,a.shape),u=Xt(c,l);let h=c,d=a;u!=null&&(d=Ue({inputs:{x:a},backend:e,attrs:{perm:u}}),h=ne(h.length,l)),Ie("sum",h,d.shape.length);const[p,f]=me(d.shape,h),m=Ke(d.dtype,"int32");let g=il(e,p,m);const x=U(f),b=e.data.get(g.dataId).values,w=e.data.get(d.dataId).values;for(let y=0;y<b.length;++y){const $=y*x;let v=0;for(let N=0;N<x;++N)v+=w[$+N];b[y]=v}if(i){const y=ee(g.shape,c),$=g;g=Wt({inputs:{x:g},backend:e,attrs:{shape:y}}),e.disposeIntermediateTensorInfo($)}return e.disposeIntermediateTensorInfo(a),u!=null&&e.disposeIntermediateTensorInfo(d),g}const TO={kernelName:cc,backendName:"cpu",kernelFunc:la};function EO(n){const{inputs:t,backend:e,attrs:s}=n,{equation:o}=s,r=t,{allDims:i,summedDims:a,idDims:l}=op(o,r.length);ip(i.length,l,r);const{path:c,steps:u}=ap(a,l),h=u.length;let d=null,p=i.length;const f=[];for(let m=0;m<h;++m){for(const g of u[m]){const{permutationIndices:x,expandDims:b}=rp(p,l[g]);let w;lp(x)?w=r[g]:(w=Ue({inputs:{x:r[g]},backend:e,attrs:{perm:x}}),f.push(w));const y=w.shape.slice();for(let $=0;$<b.length;++$)y.splice(b[$],0,1);Tt(w.shape,y)||(w=Wt({inputs:{x:w},backend:e,attrs:{shape:y}}),f.push(w)),d===null?d=w:(d=Gc({inputs:{a:w,b:d},backend:e}),f.push(d))}m<h-1&&(c[m]>=0&&(d=la({inputs:{x:d},backend:e,attrs:{axis:c[m]-(i.length-p),keepDims:!1}}),f.push(d)),p--)}for(const m of f)m!==d&&e.disposeIntermediateTensorInfo(m);return d}const RO={kernelName:wh,backendName:"cpu",kernelFunc:EO};function AO(n){const{inputs:t,backend:e}=n,{dy:s,y:o}=t;at([s,o],"eluGrad");const r=new Float32Array(U(o.shape)),i=e.data.get(o.dataId).values,a=e.data.get(s.dataId).values;for(let l=0;l<i.length;++l){const c=i[l];c>=0?r[l]=a[l]:r[l]=a[l]*(c+1)}return e.makeTensorInfo(o.shape,"float32",r)}const DO={kernelName:$h,backendName:"cpu",kernelFunc:AO};const FO=Zd,_O=Jd,OO=Qd,LO=tp,MO=ep,PO=np,zO=Dt(ci,n=>{const t=Math.sign(n),e=Math.abs(n),s=1/(1+FO*e);return t*(1-((((PO*s+MO)*s+LO)*s+OO)*s+_O)*s*Math.exp(-e*e))}),BO={kernelName:ci,backendName:"cpu",kernelFunc:zO};function cl(n){const{inputs:t,backend:e,attrs:s}=n,{input:o}=t,{dim:r}=s,i=o.shape.length,a=o.shape.slice();let l=r;return r<0&&(I(-(i+1)<=r,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+r+1),a.splice(l,0,1),Wt({inputs:{x:o},backend:e,attrs:{shape:a}})}const VO={kernelName:Dl,backendName:"cpu",kernelFunc:cl};const WO=oe((n,t)=>n/t),Wp=ue(ai,WO),Vu={kernelName:ai,backendName:"cpu",kernelFunc:Wp};function A1(n,t,e){const s=n.shape,o=s[0],r=s[1],i=e.data.get(n.dataId),a=i.complexTensorInfos.real,l=i.complexTensorInfos.imag,c=[o,r],u=U(c),h=we("float32",u),d=we("float32",u);for(let g=0;g<o;g++){const x=wo({inputs:{x:a},backend:e,attrs:{begin:[g,0],size:[1,r]}}),b=wo({inputs:{x:l},backend:e,attrs:{begin:[g,0],size:[1,r]}}),w=qe({inputs:{real:x,imag:b},backend:e}),{real:y,imag:$}=UO(w,t,e),v=os(y,$);for(let N=0;N<r;N++){const T=sp(v,N);h[g*r+N]=T.real,d[g*r+N]=T.imag}e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(b),e.disposeIntermediateTensorInfo(w)}const p=e.makeTensorInfo(c,"float32",h),f=e.makeTensorInfo(c,"float32",d),m=qe({inputs:{real:p,imag:f},backend:e});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),m}function UO(n,t,e){const s=U(n.shape),o=e.data.get(n.dataId),r=e.data.get(o.complexTensorInfos.real.dataId).values,i=e.data.get(o.complexTensorInfos.imag.dataId).values;if(GO(s)){const a=Wu(r,i,s,t,e),l=[n.shape[0],n.shape[1]];if(t){const c=e.makeTensorInfo(l,"float32",a.real),u=e.makeTensorInfo(l,"float32",a.imag),h=e.makeTensorInfo([],"float32",as(s,"float32")),d=jn({inputs:{x:h},backend:e}),p=Vu.kernelFunc({inputs:{a:c,b:h},backend:e}),f=Vu.kernelFunc({inputs:{a:u,b:d},backend:e}),m=e.data.get(p.dataId).values,g=e.data.get(f.dataId).values;return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),{real:m,imag:g}}return a}else{const a=os(r,i),l=HO(a,s,t);return Mx(l)}}function GO(n){return(n&n-1)===0}function Wu(n,t,e,s,o){if(e===1)return{real:n,imag:t};const r=os(n,t),i=e/2,a=Px(r),l=a.real,c=a.imag,u=[l.length],h=o.makeTensorInfo(u,"float32",l),d=o.makeTensorInfo(u,"float32",c),p=qe({inputs:{real:h,imag:d},backend:o}),f=zx(r),m=f.real,g=f.imag,x=[m.length],b=o.makeTensorInfo(x,"float32",m),w=o.makeTensorInfo(x,"float32",g),y=qe({inputs:{real:b,imag:w},backend:o}),$=Wu(l,c,i,s,o),v=$.real,N=$.imag,T=[v.length],k=o.makeTensorInfo(T,"float32",v),S=o.makeTensorInfo(T,"float32",N),C=qe({inputs:{real:k,imag:S},backend:o}),D=Wu(m,g,i,s,o),O=D.real,P=D.imag,M=[O.length],V=o.makeTensorInfo(M,"float32",O),H=o.makeTensorInfo(M,"float32",P),W=qe({inputs:{real:V,imag:H},backend:o}),q=Vx(e,s),j=[q.real.length],K=o.makeTensorInfo(j,"float32",q.real),Y=o.makeTensorInfo(j,"float32",q.imag),Z=qe({inputs:{real:K,imag:Y},backend:o}),tt=Gc({inputs:{a:Z,b:W},backend:o}),Q=Yo({inputs:{a:C,b:tt},backend:o}),ot=Bp({inputs:{a:C,b:tt},backend:o}),ht=yo({inputs:{input:Q},backend:o}),pt=yo({inputs:{input:ot},backend:o}),dt=Zo({inputs:{input:Q},backend:o}),bt=Zo({inputs:{input:ot},backend:o}),wt=Jo({inputs:[ht,pt],backend:o,attrs:{axis:0}}),Rt=Jo({inputs:[dt,bt],backend:o,attrs:{axis:0}}),Pt=o.data.get(wt.dataId).values,Yt=o.data.get(Rt.dataId).values;return o.disposeIntermediateTensorInfo(h),o.disposeIntermediateTensorInfo(d),o.disposeIntermediateTensorInfo(p),o.disposeIntermediateTensorInfo(b),o.disposeIntermediateTensorInfo(w),o.disposeIntermediateTensorInfo(y),o.disposeIntermediateTensorInfo(k),o.disposeIntermediateTensorInfo(S),o.disposeIntermediateTensorInfo(C),o.disposeIntermediateTensorInfo(V),o.disposeIntermediateTensorInfo(H),o.disposeIntermediateTensorInfo(W),o.disposeIntermediateTensorInfo(K),o.disposeIntermediateTensorInfo(Y),o.disposeIntermediateTensorInfo(Z),o.disposeIntermediateTensorInfo(tt),o.disposeIntermediateTensorInfo(Q),o.disposeIntermediateTensorInfo(ot),o.disposeIntermediateTensorInfo(ht),o.disposeIntermediateTensorInfo(dt),o.disposeIntermediateTensorInfo(pt),o.disposeIntermediateTensorInfo(bt),o.disposeIntermediateTensorInfo(wt),o.disposeIntermediateTensorInfo(Rt),{real:Pt,imag:Yt}}function HO(n,t,e){const s=new Float32Array(t*2);for(let o=0;o<t;o++){let r=0,i=0;for(let a=0;a<t;a++){const l=Wx(o*a,t,e),c=sp(n,a);r+=c.real*l.real-c.imag*l.imag,i+=c.real*l.imag+c.imag*l.real}e&&(r/=t,i/=t),Bx(s,r,i,o)}return s}function qO(n){const{inputs:t,backend:e}=n,{input:s}=t,o=U(s.shape),r=s.shape[s.shape.length-1],i=o/r,a=Wt({inputs:{x:s},backend:e,attrs:{shape:[i,r]}}),l=A1(a,!1,e),c=Wt({inputs:{x:l},backend:e,attrs:{shape:s.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(l),c}const jO={kernelName:Ch,backendName:"cpu",kernelFunc:qO};function Up(n){const{backend:t,attrs:e}=n,{shape:s,value:o,dtype:r}=e,i=r||vo(o),a=Jt(i,U(s));return XO(a,o,i),t.makeTensorInfo(s,i,a)}const KO={kernelName:Ih,backendName:"cpu",kernelFunc:Up};function XO(n,t,e){n.fill(t)}const YO={kernelName:vh,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,o=e,r=we(s.dtype,U(s.shape)),[i,a,l,c]=s.shape,u=o.data.get(s.dataId).values;for(let d=0;d<i;d++){const p=d*l*a*c;for(let f=0;f<a;f++){const m=f*(l*c);for(let g=0;g<l;g++){const x=g*c;for(let b=0;b<c;b++){const w=Math.round(l-g-1),y=p+m+x+b;let $=u[y];if(w>=0&&w<l){const v=w*c,N=p+m+v+b;$=u[N]}r[y]=$}}}}return{dataId:o.write(r,s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};function ZO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s;let m=E1({inputs:{x:o,filter:r},backend:e,attrs:{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d}});if(i){const g=m;if(u==="NCHW"&&i.shape.length===1&&i.shape[0]!==1){const x=Wt({inputs:{x:i},backend:e,attrs:{shape:[i.shape[0],1,1]}});m=Yo({inputs:{a:m,b:x},backend:e}),e.disposeIntermediateTensorInfo(x)}else m=Yo({inputs:{a:m,b:i},backend:e});e.disposeIntermediateTensorInfo(g)}if(p){const g=m;if(u==="NCHW"&&p==="prelu"&&a.shape.length===1&&a.shape[0]!==1){const x=Wt({inputs:{x:a},backend:e,attrs:{shape:[a.shape[0],1,1]}});m=ll(e,m,p,x,f),e.disposeIntermediateTensorInfo(x)}else m=ll(e,m,p,a,f);e.disposeIntermediateTensorInfo(g)}return m}const JO={kernelName:Ua,backendName:"cpu",kernelFunc:ZO};function QO(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s;let m=R1({inputs:{x:o,filter:r},backend:e,attrs:{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d}});if(i){const g=m;m=Yo({inputs:{a:m,b:i},backend:e}),e.disposeIntermediateTensorInfo(g)}if(p){const g=m;m=ll(e,m,p,a,f),e.disposeIntermediateTensorInfo(g)}return m}const tL={kernelName:dg,backendName:"cpu",kernelFunc:QO};function eL(n){const{inputs:t,backend:e}=n,{params:s,indices:o}=t,r=U(s.shape),i=o.shape,a=i[i.length-1],[l,c,u,h]=Sc(s,o);if(c===0)return e.makeTensorInfo(l,s.dtype,[]);const d=e.data.get(o.dataId).values,p=e.bufferSync(s),f=Vy(d,p,s.dtype,c,a,u,h,s.shape,r);return e.makeTensorInfo(l,s.dtype,f.values)}const nL={kernelName:kh,backendName:"cpu",kernelFunc:eL};function sL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,indices:r}=t,{axis:i,batchDims:a}=s;at([o,r],"gatherV2");const l=$t(i,o.shape)[0],c=e.data.get(r.dataId).values,u=o.shape[l];for(let y=0;y<c.length;++y){const $=c[y];I($<=u-1&&$>=0,()=>`GatherV2: the index value ${$} is not in [0, ${u-1}]`)}let h=a;a==null&&(h=0);const d=U(r.shape),p=up(o,r,l,h),f=Wt({inputs:{x:o},backend:e,attrs:{shape:[p.batchSize,p.outerSize,p.dimSize,p.sliceSize]}}),m=Wt({inputs:{x:r},backend:e,attrs:{shape:[p.batchSize,d/p.batchSize]}}),g=[p.batchSize,p.outerSize,d/p.batchSize,p.sliceSize],x=e.bufferSync(m),b=e.bufferSync(f),w=Wy(b,x,g);return e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),e.makeTensorInfo(p.outputShape,w.dtype,w.values)}const oL={kernelName:_l,backendName:"cpu",kernelFunc:sL};function rL(n){const{inputs:t,backend:e}=n,{input:s}=t,o=U(s.shape),r=s.shape[s.shape.length-1],i=o/r,a=Wt({inputs:{x:s},backend:e,attrs:{shape:[i,r]}}),l=A1(a,!0,e),c=Wt({inputs:{x:l},backend:e,attrs:{shape:s.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(l),c}const iL={kernelName:Sh,backendName:"cpu",kernelFunc:rL};const aL=Dt(gi,n=>Number.isFinite(n)?1:0,"bool"),lL={kernelName:gi,backendName:"cpu",kernelFunc:aL};const cL=Dt(xi,n=>Math.abs(n)===1/0?1:0,"bool"),uL={kernelName:xi,backendName:"cpu",kernelFunc:cL};const hL=Dt(bi,n=>Number.isNaN(n)?1:0,"bool"),dL={kernelName:bi,backendName:"cpu",kernelFunc:hL};function pL(n){const{backend:t,attrs:e}=n,{start:s,stop:o,num:r}=e,i=jy(s,o,r);return t.makeTensorInfo([i.length],"float32",i)}const fL={kernelName:ng,backendName:"cpu",kernelFunc:pL};const mL=Dt(wi,n=>Math.log1p(n)),gL={kernelName:wi,backendName:"cpu",kernelFunc:mL};const xL=oe((n,t)=>n&&t),bL=ue(zl,xL,null,"bool"),yL={kernelName:zl,backendName:"cpu",kernelFunc:bL};const wL=Dt(Bl,n=>n?0:1,"bool"),$L={kernelName:Bl,backendName:"cpu",kernelFunc:wL};const CL=oe((n,t)=>n||t),IL=ue(Vl,CL,null,"bool"),vL={kernelName:Vl,backendName:"cpu",kernelFunc:IL};function kL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{depthRadius:r,bias:i,alpha:a,beta:l}=s;at(o,"LRN");const c=o.shape[3],u=c-1,h=e.data.get(o.dataId).values,d=U(o.shape),p=new Float32Array(d);function f(m){const g=m%c;let x=m-g+Math.max(0,g-r);const b=m-g+Math.min(g+r,u);let w=0;for(;x<=b;x++){const y=h[x];w+=y*y}return w}for(let m=0;m<d;m++){const g=f(m),x=h[m]*Math.pow(i+a*g,-l);p[m]=x}return e.makeTensorInfo(o.shape,o.dtype,p)}const SL={kernelName:Wl,backendName:"cpu",kernelFunc:kL};function NL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,y:r,dy:i}=t,{depthRadius:a,bias:l,alpha:c,beta:u}=s;at(i,"LRNGrad");const h=U(i.shape),d=i.shape[3],p=e.data.get(i.dataId).values,f=e.data.get(o.dataId).values,m=e.data.get(r.dataId).values,g=new Float32Array(h),x=h;for(let b=0;b<x;b++){const w=b%d,y=b-w+Math.max(0,w-a),$=b-w+Math.min(d,w+a+1);let v=0;for(let N=y;N<$;N++)v+=Math.pow(f[N],2);v=c*v+l;for(let N=y;N<$;N++){let T=-2*c*u*f[N]*m[b]/v;b===N&&(T+=Math.pow(v,-u)),T*=p[b],g[N]+=T}}return e.makeTensorInfo(i.shape,o.dtype,g)}const TL={kernelName:Th,backendName:"cpu",kernelFunc:NL};function D1(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reductionIndices:r,keepDims:i}=s,a=e;let l=o.shape;const c=l.length,u=$t(r,l);let h=u;const d=Xt(h,c);let p=a.data.get(o.dataId).values;if(d!=null){const y=new Array(c);for(let $=0;$<y.length;$++)y[$]=l[d[$]];p=Pp(p,l,o.dtype,d,y),h=ne(h.length,c),l=y}at(o,"max"),Ie("max",h,c);const[f,m]=me(l,h),g=U(m),x=Xy(p,g,f,o.dtype),b=a.write(x,f,o.dtype);let w=f;return i&&(w=ee(f,u)),{dataId:b,shape:w,dtype:o.dtype}}const EL={kernelName:Ul,backendName:"cpu",kernelFunc:D1};function RL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;at(o,"maxPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;I(Te(i,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=xn(o.shape,r,i,c,a,l);let h;if(u.filterWidth===1&&u.filterHeight===1&&Tt(u.inShape,u.outShape))h=jn({inputs:{x:o},backend:e});else{const d=e.data.get(o.dataId).values,p=ut(o.shape),f=Vp(d,o.shape,o.dtype,p,u,"max");h=e.makeTensorInfo(u.outShape,o.dtype,f.values)}return h}const AL={kernelName:Gl,backendName:"cpu",kernelFunc:RL};function DL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s;at(o,"maxPool3d");const u=cs(o.shape,r,i,1,a,l,c),h=e.data.get(o.dataId).values,d=T1(h,o.shape,o.dtype,ut(o.shape),u,"max");return e.makeTensorInfo(d.shape,"float32",d.values)}const FL={kernelName:Hl,backendName:"cpu",kernelFunc:DL};function _L(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,{filterSize:i,strides:a,pad:l,dimRoundingMode:c}=s;at([o,r],"maxPool3DGrad");const u=cs(r.shape,i,a,1,l,c),h=e.bufferSync(r),d=C_(h,u),p=u.strideDepth,f=u.strideHeight,m=u.strideWidth,g=u.dilationDepth,x=u.dilationHeight,b=u.dilationWidth,w=u.effectiveFilterDepth,y=u.effectiveFilterHeight,$=u.effectiveFilterWidth,v=w-1-u.padInfo.front,N=$-1-u.padInfo.left,T=y-1-u.padInfo.top,k=Ct(r.shape,"float32"),S=e.bufferSync(o);for(let C=0;C<u.batchSize;++C)for(let D=0;D<u.inChannels;++D)for(let O=0;O<u.inDepth;++O)for(let P=0;P<u.inHeight;++P)for(let M=0;M<u.inWidth;++M){const V=O-v,H=P-T,W=M-N;let q=0;for(let j=0;j<w;j+=g){const K=(V+j)/p;if(!(K<0||K>=u.outDepth||Math.floor(K)!==K))for(let Y=0;Y<y;Y+=x){const Z=(H+Y)/f;if(!(Z<0||Z>=u.outHeight||Math.floor(Z)!==Z))for(let tt=0;tt<$;tt+=b){const Q=(W+tt)/m;if(Q<0||Q>=u.outWidth||Math.floor(Q)!==Q)continue;const ot=w*y*$-1-d.get(C,K,Z,Q,D),ht=j*y*$+Y*$+tt,pt=ot===ht?1:0;if(pt===0)continue;const dt=S.get(C,K,Z,Q,D);q+=dt*pt}}}k.set(q,C,O,P,M,D)}return e.makeTensorInfo(k.shape,k.dtype,k.values)}const OL={kernelName:Rh,backendName:"cpu",kernelFunc:_L};function LL(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r,output:i}=t,a=r;at([r,i],"maxPoolGrad");const{filterSize:l,strides:c,pad:u,dimRoundingMode:h}=s,d=xn(a.shape,l,c,1,u,h),p=e.data.get(a.dataId).values,f=Ct(d.outShape,a.dtype,N1(p,a.shape,a.dtype,d).values),m=d.strideHeight,g=d.strideWidth,x=d.dilationHeight,b=d.dilationWidth,w=d.effectiveFilterHeight,y=d.effectiveFilterWidth,$=y-1-d.padInfo.left,v=w-1-d.padInfo.top,N=Ct(a.shape,"float32"),T=e.data.get(o.dataId).values,k=Ct(o.shape,"float32",T);for(let S=0;S<d.batchSize;++S)for(let C=0;C<d.inChannels;++C)for(let D=0;D<d.inHeight;++D)for(let O=0;O<d.inWidth;++O){const P=D-v,M=O-$;let V=0;for(let H=0;H<w;H+=x){const W=(P+H)/m;if(!(W<0||W>=d.outHeight||Math.floor(W)!==W))for(let q=0;q<y;q+=b){const j=(M+q)/g;if(j<0||j>=d.outWidth||Math.floor(j)!==j)continue;const K=w*y-1-f.get(S,W,j,C),Y=H*y+q,Z=K===Y?1:0;if(Z===0)continue;const tt=k.get(S,W,j,C);V+=tt*Z}}N.set(V,S,D,O,C)}return e.makeTensorInfo(N.shape,N.dtype,N.values)}const ML={kernelName:Eh,backendName:"cpu",kernelFunc:LL};function PL(n,t,e,s,o){const r=ut(t),i=Vp(n,t,e,r,o,"max"),a=N1(n,t,e,o,!0,s);return[i.values,a.values]}const zL={kernelName:sg,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{filterSize:o,strides:r,pad:i,includeBatchInIndex:a}=t,l=e;at(s,"MaxPoolWithArgmax");const c=l.data.get(s.dataId).values,u=xn(s.shape,o,r,[1,1],i),[h,d]=PL(c,s.shape,s.dtype,a,u),p=l.write(h,u.outShape,s.dtype),f=l.write(d,u.outShape,s.dtype);return[{dataId:p,shape:u.outShape,dtype:s.dtype},{dataId:f,shape:u.outShape,dtype:"int32"}]}};function BL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=$t(r,o.shape),c=me(o.shape,a)[1],u=U(c),h=[],d=e.makeTensorInfo([],"float32",new Float32Array([u]));h.push(d);const p=Os({inputs:{x:o},backend:e,attrs:{dtype:"float32"}});h.push(p);const f=Wp({inputs:{a:p,b:d},backend:e});h.push(f);const m=la({inputs:{x:f},backend:e,attrs:{axis:r,keepDims:i}});return h.forEach(g=>e.disposeIntermediateTensorInfo(g)),m}const VL={kernelName:ql,backendName:"cpu",kernelFunc:BL};function WL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;at(o,"min");const a=$t(r,o.shape);let l=a;const c=Xt(l,o.shape.length);let u=o;c!=null&&(u=Ue({inputs:{x:o},backend:e,attrs:{perm:c}}),l=ne(l.length,o.shape.length)),Ie("min",l,u.shape.length);const[h,d]=me(u.shape,l),p=U(d),f=$e(U(h),u.dtype),m=e.data.get(u.dataId).values;for(let x=0;x<f.length;++x){const b=x*p;let w=m[b];for(let y=0;y<p;++y){const $=m[b+y];(Number.isNaN($)||$<w)&&(w=$)}f[x]=w}c!=null&&e.disposeIntermediateTensorInfo(u);const g=e.makeTensorInfo(h,u.dtype,f);if(i){const x=ee(h,a),b=Wt({inputs:{x:g},backend:e,attrs:{shape:x}});return e.disposeIntermediateTensorInfo(g),b}return g}const UL={kernelName:jl,backendName:"cpu",kernelFunc:WL};function GL(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,mode:i}=s;at(o,"mirrorPad");const a=r.map((w,y)=>w[0]+o.shape[y]+w[1]),l=r.map(w=>w[0]),c=r.map((w,y)=>w[0]+o.shape[y]),u=i==="reflect"?0:1,h=e.data.get(o.dataId).values,d=o.shape.length,p=ut(o.shape),f=U(a),m=a.length,g=ut(a),x=we(o.dtype,f);for(let w=0;w<f;w++){let y=ko(w,m,g);for(let v=0;v<m;v++)y[v]<l[v]?y[v]=l[v]*2-y[v]-u:y[v]>=c[v]&&(y[v]=(c[v]-1)*2-y[v]+u);y=y.map((v,N)=>v-l[N]);const $=Sn(y,d,p);x[w]=h[$]}return{dataId:e.write(x,a,o.dtype),shape:a,dtype:o.dtype}}const HL={kernelName:Kl,backendName:"cpu",kernelFunc:GL};const qL=oe(((n,t)=>{const e=n%t;return n<0&&t<0||n>=0&&t>=0?e:(e+t)%t})),jL=ue(Ii,qL),KL={kernelName:Ii,backendName:"cpu",kernelFunc:jL};function F1(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{dim:r}=s,i=o.shape.length;let a=r;if(a===-1&&(a=i-1),a!==i-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${i} and dim was ${a}`);const l=$t([a],o.shape),c=D1({inputs:{x:o},backend:e,attrs:{reductionIndices:l,keepDims:!1}}),u=ee(c.shape,l),h=Wt({inputs:{x:c},backend:e,attrs:{shape:u}}),d=Bp({inputs:{a:o,b:h},backend:e}),p=My({inputs:{x:d},backend:e}),f=la({inputs:{x:p},backend:e,attrs:{axis:l,keepDims:!1}}),m=Wt({inputs:{x:f},backend:e,attrs:{shape:u}}),g=Wp({inputs:{a:p,b:m},backend:e});return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),g}const XL={kernelName:dc,backendName:"cpu",kernelFunc:F1};function YL(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{numSamples:r,seed:i,normalized:a}=s;at(o,"multinomial");const l=a?o:F1({inputs:{logits:o},backend:e,attrs:{dim:-1}}),c=l.shape[0],u=l.shape[1],h=e.data.get(l.dataId).values,d=[c,r],p=$e(U(d),"int32");for(let f=0;f<c;++f){const m=f*u,g=new Float32Array(u-1);g[0]=h[m];for(let w=1;w<g.length;++w)g[w]=g[w-1]+h[m+w];const x=sr.alea(i.toString()),b=f*r;for(let w=0;w<r;++w){const y=x();p[b+w]=g.length;for(let $=0;$<g.length;$++)if(y<g[$]){p[b+w]=$;break}}}return a||e.disposeIntermediateTensorInfo(l),e.makeTensorInfo(d,"int32",p)}const ZL={kernelName:og,backendName:"cpu",kernelFunc:YL};const JL=Od;function QL(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l}=s;at(o,"NonMaxSuppression");const c=e.data.get(o.dataId).values,u=e.data.get(r.dataId).values,{selectedIndices:h}=JL(c,u,i,a,l);return e.makeTensorInfo([h.length],"int32",new Int32Array(h))}const tM={kernelName:Ah,backendName:"cpu",kernelFunc:QL};const eM=Ld;function nM(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,padToMaxOutputSize:c}=s;at(o,"NonMaxSuppressionPadded");const u=e.data.get(o.dataId).values,h=e.data.get(r.dataId).values,{selectedIndices:d,validOutputs:p}=eM(u,h,i,a,l,c);return[e.makeTensorInfo([d.length],"int32",new Int32Array(d)),e.makeTensorInfo([],"int32",new Int32Array([p]))]}const sM={kernelName:Dh,backendName:"cpu",kernelFunc:nM};const oM=Md;function rM(n){const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,softNmsSigma:c}=s;at(o,"NonMaxSuppressionWithScore");const u=e.data.get(o.dataId).values,h=e.data.get(r.dataId).values,d=i,p=a,f=l,m=c,{selectedIndices:g,selectedScores:x}=oM(u,h,d,p,f,m);return[e.makeTensorInfo([g.length],"int32",new Int32Array(g)),e.makeTensorInfo([x.length],"float32",new Float32Array(x))]}const iM={kernelName:Fh,backendName:"cpu",kernelFunc:rM};function aM(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o}=t,{dtype:r,depth:i,onValue:a,offValue:l}=s;at(o,"oneHot");const c=U(o.shape),u=new Float32Array(c*i);u.fill(l);const h=e.data.get(o.dataId).values;for(let d=0;d<c;++d)h[d]>=0&&h[d]<i&&(u[d*i+h[d]]=a);return e.makeTensorInfo([...o.shape,i],r,u)}const lM={kernelName:Jl,backendName:"cpu",kernelFunc:aM};function ul(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("zerosLike is not supported for string tensors");if(s.dtype==="complex64"){const o=yo({inputs:{input:s},backend:e}),r=ul({inputs:{x:o},backend:e}),i=Zo({inputs:{input:s},backend:e}),a=ul({inputs:{x:i},backend:e}),l=qe({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return Up({backend:e,attrs:{shape:s.shape,value:0,dtype:s.dtype}})}const cM={kernelName:gc,backendName:"cpu",kernelFunc:ul};function _1(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("onesLike is not supported for string tensors");if(s.dtype==="complex64"){const o=yo({inputs:{input:s},backend:e}),r=_1({inputs:{x:o},backend:e}),i=Zo({inputs:{input:s},backend:e}),a=ul({inputs:{x:i},backend:e}),l=qe({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return Up({backend:e,attrs:{shape:s.shape,value:1,dtype:s.dtype}})}const uM={kernelName:Zl,backendName:"cpu",kernelFunc:_1};function O1(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s;if(t.length===1)return cl({inputs:{input:t[0]},backend:e,attrs:{dim:o}});const r=t[0].shape,i=t[0].dtype;t.forEach(u=>{Oe(r,u.shape,"All tensors passed to stack must have matching shapes"),I(i===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=[],l=t.map(u=>{const h=cl({inputs:{input:u},backend:e,attrs:{dim:o}});return a.push(h),h}),c=Jo({inputs:l,backend:e,attrs:{axis:o}});return a.forEach(u=>e.disposeIntermediateTensorInfo(u)),c}const hM={kernelName:Ql,backendName:"cpu",kernelFunc:O1};function dM(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,constantValue:i}=s;at(o,"pad");const a=r.map((b,w)=>b[0]+o.shape[w]+b[1]),l=r.map(b=>b[0]),c=e.data.get(o.dataId).values,u=U(o.shape),h=o.shape.length,d=ut(o.shape),p=U(a),f=a.length,m=ut(a),g=we(o.dtype,p);i!==0&&g.fill(i);for(let b=0;b<u;b++){const y=ko(b,h,d).map((v,N)=>v+l[N]),$=Sn(y,f,m);g[$]=c[b]}return{dataId:e.write(g,a,o.dtype),shape:a,dtype:o.dtype}}const L1={kernelName:tc,backendName:"cpu",kernelFunc:dM};const pM=oe((n,t)=>Math.pow(n,t)),fM=ue(ki,pM),mM={kernelName:ki,backendName:"cpu",kernelFunc:fM};function gM(n){const{inputs:t,backend:e,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:i}=t,{outputRaggedRank:a}=s,l=o.map(x=>e.data.get(x.dataId).values),c=o.map(x=>x.shape),u=e.data.get(r.dataId).values,h=e.data.get(i.dataId).values,[d,p,f]=e1(l,c,u,r.shape,r.dtype,h,i.shape),m=d.map(x=>e.makeTensorInfo([x.length],"int32",x)),g=e.makeTensorInfo(f,r.dtype,p);return m.concat([g])}const xM={kernelName:rg,backendName:"cpu",kernelFunc:gM};function bM(n){const{inputs:t,backend:e}=n,{starts:s,limits:o,deltas:r}=t,i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=n1(i,s.shape,s.dtype,a,o.shape,l,r.shape),h=e.makeTensorInfo([c.length],"int32",c),d=e.makeTensorInfo([u.length],s.dtype,u);return[h,d]}const yM={kernelName:ig,backendName:"cpu",kernelFunc:bM};function wM(n){const{inputs:t,backend:e,attrs:s}=n,{shape:o,values:r,defaultValue:i,rowPartitionTensors:a}=t,{rowPartitionTypes:l}=s,c=e.data.get(o.dataId).values,u=e.data.get(r.dataId).values,h=e.data.get(i.dataId).values,d=a.map(g=>e.data.get(g.dataId).values),p=a.map(g=>g.shape),[f,m]=s1(c,o.shape,u,r.shape,r.dtype,h,i.shape,d,p,l);return e.makeTensorInfo(f,r.dtype,m)}const $M={kernelName:ag,backendName:"cpu",kernelFunc:wM};function CM(n){const{backend:t,attrs:e}=n,{start:s,stop:o,dtype:r,step:i}=e,a=o1(s,o,i,r);return t.makeTensorInfo([a.length],r,a)}const IM={kernelName:_h,backendName:"cpu",kernelFunc:CM};const vM=Dt(Si,n=>1/n),kM={kernelName:Si,backendName:"cpu",kernelFunc:vM};function SM(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s;at(o,"resizeBilinear");const l=ut(o.shape),[c,u]=a,[h,d,p,f]=o.shape,m=e.data.get(o.dataId).values,g=new Float32Array(U([h,c,u,f])),x=[r&&c>1?d-1:d,r&&u>1?p-1:p],b=[r&&c>1?c-1:c,r&&u>1?u-1:u];let w=0;const y=x[0]/b[0],$=x[1]/b[1];for(let v=0;v<h;v++)for(let N=0;N<c;N++){let T;i?T=y*(N+.5)-.5:T=y*N;const k=Math.max(0,Math.floor(T)),S=T-k,C=Math.min(d-1,Math.ceil(T)),D=v*l[0]+k*l[1],O=v*l[0]+C*l[1];for(let P=0;P<u;P++){let M;i?M=$*(P+.5)-.5:M=$*P;const V=Math.max(0,Math.floor(M)),H=M-V,W=Math.min(p-1,Math.ceil(M)),q=D+V*l[2],j=O+V*l[2],K=D+W*l[2],Y=O+W*l[2];for(let Z=0;Z<f;Z++){const tt=m[q+Z],Q=m[j+Z],ot=m[K+Z],ht=m[Y+Z],pt=tt+(ot-tt)*H,dt=Q+(ht-Q)*H,bt=pt+(dt-pt)*S;g[w++]=bt}}}return e.makeTensorInfo([h,c,u,f],"float32",g)}const NM={kernelName:rc,backendName:"cpu",kernelFunc:SM};function TM(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s;at([r,o],"resizeBilinearGrad");const a=ut(o.shape),[l,c,u,h]=o.shape,[,d,p]=r.shape,f=new Float32Array(l*c*u*h),m=[i&&d>1?c-1:c,i&&p>1?u-1:u],g=[i&&d>1?d-1:d,i&&p>1?p-1:p],x=m[0]/g[0],b=m[1]/g[1],w=e.data.get(r.dataId).values;let y=0;for(let $=0;$<l;$++){const v=$*a[0];for(let N=0;N<d;N++){const T=N*x,k=Math.floor(T),S=Math.min(Math.ceil(T),c-1),C=v+k*a[1],D=v+S*a[1],O=T-k,P=1-O;for(let M=0;M<p;M++){const V=M*b,H=Math.floor(V),W=Math.min(Math.ceil(V),u-1),q=V-H,j=1-q,K=C+H*a[2],Y=C+W*a[2],Z=D+H*a[2],tt=D+W*a[2],Q=P*j,ot=P*q,ht=O*j,pt=O*q;for(let dt=0;dt<h;dt++){const bt=w[y++];f[K+dt]+=bt*Q,f[Y+dt]+=bt*ot,f[Z+dt]+=bt*ht,f[tt+dt]+=bt*pt}}}}return e.makeTensorInfo([l,u,c,h],"float32",f)}const EM={kernelName:Mh,backendName:"cpu",kernelFunc:TM};function RM(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s;at(o,"resizeNearestNeighbor");const l=ut(o.shape),[c,u]=a,[h,d,p,f]=o.shape,m=e.data.get(o.dataId).values,g=new Float32Array(h*c*u*f),x=[r&&c>1?d-1:d,r&&u>1?p-1:p],b=[r&&c>1?c-1:c,r&&u>1?u-1:u],w=x[0]/b[0],y=x[1]/b[1];let $=0;for(let v=0;v<h;v++){const N=v*l[0];for(let T=0;T<c;T++){const k=i?w*(T+.5):w*T;let S=Math.min(d-1,r?Math.round(k):Math.floor(k));i&&(S=Math.max(0,S));const C=N+S*l[1];for(let D=0;D<u;D++){const O=i?y*(D+.5):y*D;let P=Math.min(p-1,r?Math.round(O):Math.floor(O));i&&(P=Math.max(0,P));const M=C+P*l[2];for(let V=0;V<f;V++){const H=m[M+V];g[$++]=H}}}}return e.makeTensorInfo([h,c,u,f],o.dtype,g)}const AM={kernelName:oc,backendName:"cpu",kernelFunc:RM};function DM(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s;at([r,o],"resizeNearestNeighborGrad");const a=ut(o.shape),l=ut(r.shape),[c,u,h,d]=o.shape,[,p,f]=r.shape,m=new Float32Array(c*u*h*d),g=e.data.get(r.dataId).values,x=[i&&p>1?u-1:u,i&&f>1?h-1:h],b=[i&&p>1?p-1:p,i&&f>1?f-1:f],w=x[0]/b[0],y=x[1]/b[1],$=1/w,v=1/y,N=Math.ceil($)*2+2,T=Math.ceil(v)*2+2;for(let k=0;k<c;k++){const S=k*a[0];for(let C=0;C<u;C++){const D=S+C*a[1],O=Math.floor(C*$),P=Math.floor(O-N/2);for(let M=0;M<h;M++){const V=D+M*a[2],H=Math.floor(M*v),W=Math.floor(H-T/2);for(let q=0;q<d;q++){let j=0;for(let K=0;K<N;K++){const Y=K+P;if(Y<0||Y>=p)continue;const Z=S+Y*l[1],tt=Y*w,Q=Math.min(u-1,i?Math.round(tt):Math.floor(tt));if(C===Q)for(let ot=0;ot<T;ot++){const ht=ot+W;if(ht<0||ht>=f)continue;const pt=Z+ht*l[2],dt=ht*y,bt=Math.min(h-1,i?Math.round(dt):Math.floor(dt));M===bt&&(j+=g[pt+q])}}m[V+q]=j}}}}return e.makeTensorInfo(o.shape,o.dtype,m)}const FM={kernelName:Lh,backendName:"cpu",kernelFunc:DM};function _M(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dims:r}=s;at(o,"reverse");const i=o.shape.length,a=$t(r,o.shape);if(i===0)return jn({inputs:{x:o},backend:e});const l=new ye(o.shape,o.dtype),c=e.bufferSync(o);for(let u=0;u<l.size;u++){const h=l.indexToLoc(u),d=h.slice();a.forEach(p=>d[p]=o.shape[p]-1-d[p]),l.set(c.get(...d),...h)}return e.makeTensorInfo(l.shape,l.dtype,l.values)}const OM={kernelName:ic,backendName:"cpu",kernelFunc:_M};const LM={kernelName:Yh,backendName:"cpu",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,{radians:o,fillValue:r,center:i}=t,a=e,l=we(s.dtype,U(s.shape)),[c,u,h,d]=s.shape,[p,f]=Kd(i,u,h),m=255,g=Math.sin(o),x=Math.cos(o),b=a.data.get(s.dataId).values;for(let y=0;y<c;y++){const $=y*h*u*d;for(let v=0;v<u;v++){const N=v*(h*d);for(let T=0;T<h;T++){const k=T*d;for(let S=0;S<d;S++){const C=[c,v,T,S],D=C[2],O=C[1];let P=(D-p)*x-(O-f)*g,M=(D-p)*g+(O-f)*x;P=Math.round(P+p),M=Math.round(M+f);let V=r;if(typeof r!="number"&&(S===3?V=m:V=r[S]),P>=0&&P<h&&M>=0&&M<u){const W=M*(h*d),q=P*d,j=$+W+q+S;V=b[j]}const H=$+N+k+S;l[H]=V}}}}return{dataId:a.write(l,s.shape,s.dtype),shape:s.shape,dtype:s.dtype}}};const MM=Dt(Ei,n=>{const t=Math.floor(n);return n-t<.5?Math.floor(n):n-t>.5?Math.ceil(n):t%2===0?t:t+1}),PM={kernelName:Ei,backendName:"cpu",kernelFunc:MM};function zM(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o,updates:r}=t,{shape:i}=s,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Bs(r,o,i),d=!0,p=e.bufferSync(o),f=e.bufferSync(r),m=Qs(p,f,i,h,c,l,a,u,0,d);return e.makeTensorInfo(i,m.dtype,m.values)}const BM={kernelName:lg,backendName:"cpu",kernelFunc:zM};function VM(n,t){let e=0,s=n.length,o=0;for(;e<s;)o=Math.floor((e+s)/2),n[o]<t?e=o+1:s=o;return s}function WM(n,t){let e=0,s=n.length,o=0;for(;e<s;)o=Math.floor((e+s)/2),n[o]<=t?e=o+1:s=o;return s}function UM(n,t,e,s,o,r){const i=Jt("int32",e*o);for(let a=0;a<e;++a){const l=n.slice(a*s,(a+1)*s),c=a*o;for(let u=0;u<o;++u)i[c+u]=r==="left"?VM(l,t[u+c]):WM(l,t[u+c])}return i}function GM(n){const{inputs:t,backend:e,attrs:s}=n,{sortedSequence:o,values:r}=t,{side:i}=s,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,c=UM(a,l,o.shape[0],o.shape[1],r.shape[1],i);return e.makeTensorInfo(r.shape,"int32",c)}const HM={kernelName:ug,backendName:"cpu",kernelFunc:GM};function qM(n){const{inputs:t,backend:e}=n,{condition:s,t:o,e:r}=t;at([s,o,r],"select");const i=s.shape.length,a=e.data.get(s.dataId).values,l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=Ke(o.dtype,r.dtype),h=$e(U(o.shape),u);let d=0;const p=i===0||i>1||o.shape.length===1?1:U(o.shape.slice(1));for(let f=0;f<a.length;f++)for(let m=0;m<p;m++)a[f]===1?h[d++]=l[f]:h[d++]=c[f];return e.makeTensorInfo(o.shape,u,h)}const jM={kernelName:ac,backendName:"cpu",kernelFunc:qM};const KM=Ec,XM=Rc,YM=Dt(Ai,n=>n>=0?XM*n:KM*(Math.exp(n)-1)),ZM={kernelName:Ai,backendName:"cpu",kernelFunc:YM};const JM=Dt(_i,n=>n<0?-1:n>0?1:0),QM={kernelName:_i,backendName:"cpu",kernelFunc:JM};const tP=Dt(Di,n=>Math.sin(n)),eP={kernelName:Di,backendName:"cpu",kernelFunc:tP};const nP=Dt(Fi,n=>Math.sinh(n)),sP={kernelName:Fi,backendName:"cpu",kernelFunc:nP};const oP=11920928955078125e-23,bm=Math.log(oP)+2,rP=Dt(Li,n=>{const t=n>-bm,e=n<bm,s=Math.exp(n);let o;return e?o=s:t?o=n:o=Math.log(1+s),o}),iP={kernelName:Li,backendName:"cpu",kernelFunc:rP};function aP(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,paddings:i}=s;at([o],"spaceToBatchND");const a=U(r),l=[[0,0]];l.push(...i);for(let v=1+r.length;v<o.shape.length;++v)l.push([0,0]);const c=L1.kernelFunc({inputs:{x:o},backend:e,attrs:{paddings:l,constantValue:0}}),u=Zi(c.shape,r,a,!1),h=Ji(u.length,r.length,!1),d=Qi(c.shape,r,a,!1),m=Wt({inputs:{x:c},backend:e,attrs:{shape:u}}),b=Ue({inputs:{x:m},backend:e,attrs:{perm:h}}),$=Wt({inputs:{x:b},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(b),$}const lP={kernelName:uc,backendName:"cpu",kernelFunc:aP};function cP(n){const{inputs:t,backend:e}=n,{indices:s,values:o,denseShape:r,defaultValue:i}=t;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
        ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
        ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
        ${o.shape}`);if(i.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${i.shape}`);const a=e.data.get(s.dataId).values,l=e.data.get(o.dataId).values,c=e.data.get(r.dataId).values,u=e.data.get(i.dataId).values[0],[h,d,p,f,m]=l1(a,s.shape,s.dtype,l,o.dtype,c,u);return[e.makeTensorInfo(d,s.dtype,h),e.makeTensorInfo([d[0]],o.dtype,p),e.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(g=>Number(g)))),e.makeTensorInfo([m.length],s.dtype,new Int32Array(m))]}const uP={kernelName:Ph,backendName:"cpu",kernelFunc:cP};function hP(n){const{inputs:t,backend:e}=n,{inputIndices:s,inputShape:o,newShape:r}=t;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape
        ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape
        ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const i=Array.from(e.data.get(o.dataId).values),a=e.data.get(s.dataId).values,l=Array.from(e.data.get(r.dataId).values),[c,u,h]=c1(a,s.shape,s.dtype,i,l);return[e.makeTensorInfo(u,s.dtype,c),e.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}const dP={kernelName:zh,backendName:"cpu",kernelFunc:hP};function pP(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
          ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
          ${r.shape}`);if(o.shape[0]!==r.shape[0])throw new Error("segmentIds and indices should have same size.");const i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=zp(i,s.shape,s.dtype,a,l,!0);return e.makeTensorInfo(u,s.dtype,c)}const fP={kernelName:Bh,backendName:"cpu",kernelFunc:pP};function mP(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
         ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
         ${r.shape}`);if(o.shape[0]!==r.shape[0])throw new Error("segmentIds and indices should have same size.");const i=e.data.get(s.dataId).values,a=e.data.get(o.dataId).values,l=e.data.get(r.dataId).values,[c,u]=zp(i,s.shape,s.dtype,a,l);return e.makeTensorInfo(u,s.dtype,c)}const gP={kernelName:Vh,backendName:"cpu",kernelFunc:mP};function xP(n){const{inputs:t,backend:e,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:i}=t,{outputShape:a}=s,{sliceRank:l,numUpdates:c,sliceSize:u,strides:h,outputSize:d}=Bs(r,o,a),p=!1,f=e.bufferSync(o);let m;switch(r.dtype){case"bool":{const g=e.bufferSync(r),x=!!e.data.get(i.dataId).values[0];m=Qs(f,g,a,d,u,c,l,h,x,p);break}case"float32":{const g=e.bufferSync(r),x=e.data.get(i.dataId).values[0];m=Qs(f,g,a,d,u,c,l,h,x,p);break}case"int32":{const g=e.bufferSync(r),x=e.data.get(i.dataId).values[0];m=Qs(f,g,a,d,u,c,l,h,x,p);break}case"string":{const g=e.bufferSync(r),x=ns(e.data.get(i.dataId).values[0]);m=Qs(f,g,a,d,u,c,l,h,x,p);break}default:throw new Error(`Unsupported type ${r.dtype}`)}return e.makeTensorInfo(a,m.dtype,m.values)}const bP={kernelName:hg,backendName:"cpu",kernelFunc:xP};function yP(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{numOrSizeSplits:r,axis:i}=s,a=$t(i,o.shape)[0],l=cp(o,r,a),c=new Array(o.shape.length).fill(0),u=o.shape.slice();return l.map(h=>{const d=[...u];d[a]=h;const p=wo({inputs:{x:o},backend:e,attrs:{begin:c,size:d}});return c[a]+=h,p})}const wP={kernelName:hc,backendName:"cpu",kernelFunc:yP};const $P={kernelName:Wh,backendName:"cpu",kernelFunc:({inputs:n,backend:t})=>{const{x:e}=n,s=t;at(e,"square");const o=s.data.get(e.dataId).values,r=new Float32Array(o.length);for(let a=0;a<o.length;++a){const l=o[a];r[a]=l*l}return{dataId:s.write(r,e.shape,e.dtype),shape:e.shape,dtype:e.dtype}}};const CP=Dt(Ui,(n,t)=>{const e=t;return isNaN(n)?NaN:n>0?1:e.alpha}),IP={kernelName:Ui,backendName:"cpu",kernelFunc:CP};function vP(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,end:i,strides:a,beginMask:l,endMask:c,ellipsisMask:u,newAxisMask:h,shrinkAxisMask:d}=s;at(o,"stridedSlice");const{finalShapeSparse:p,finalShape:f,isIdentity:m,sliceDim0:g,isSimpleSlice:x,begin:b,end:w,strides:y}=Hd(o.shape,r,i,a,l,c,u,h,d);let $;if(m)$=Wt({inputs:{x:o},backend:e,attrs:{shape:f}});else if(g||x){I(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const v=Wd(b,w,y),N=wo({inputs:{x:o},backend:e,attrs:{begin:b,size:v}});$=Wt({inputs:{x:N},backend:e,attrs:{shape:f}}),e.disposeIntermediateTensorInfo(N)}else{const v=e.bufferSync(o),N=d1(p,v,y,b);$=e.makeTensorInfo(f,N.dtype,N.values)}return $}const kP={kernelName:Uh,backendName:"cpu",kernelFunc:vP};function SP(n){const{inputs:t,backend:e,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:i,rightPad:a,padWidth:l,preserveShortSequences:c}=s,{data:u,dataSplits:h}=t,d=e.data.get(u.dataId).values,p=e.data.get(h.dataId).values,[f,m]=p1(d,p,o,r,i,a,l,c);return[e.makeTensorInfo([f.length],"string",f),e.makeTensorInfo(h.shape,"int32",m)]}const NP={kernelName:Gh,backendName:"cpu",kernelFunc:SP};function TP(n){const{inputs:t,backend:e,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:i}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(i.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${i.shape}`);const a=e.data.get(r.dataId).values,l=e.data.get(i.dataId).values[0],[c,u,h]=f1(a,l,o),d=u.length;return[e.makeTensorInfo([d,2],"int32",c),e.makeTensorInfo([d],"string",u),e.makeTensorInfo([2],"int32",new Int32Array(h))]}const EP={kernelName:Hh,backendName:"cpu",kernelFunc:TP};function RP(n){const{inputs:t,backend:e,attrs:s}=n,{numBuckets:o}=s,{input:r}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const i=e.data.get(r.dataId).values,a=m1(i,o);return e.makeTensorInfo(r.shape,"int32",a)}const AP={kernelName:qh,backendName:"cpu",kernelFunc:RP};const DP=Dt(Bi,n=>Math.tan(n)),FP={kernelName:Bi,backendName:"cpu",kernelFunc:DP};const _P=Dt(Vi,n=>Math.tanh(n)),OP={kernelName:Vi,backendName:"cpu",kernelFunc:_P};function LP(n){const{inputs:t,backend:e}=n,{tensor:s,indices:o,updates:r}=t,{sliceRank:i,numUpdates:a,sliceSize:l,strides:c,outputSize:u}=Bs(r,o,s.shape),h=!1,d=e.bufferSync(o),p=e.bufferSync(r),f=e.bufferSync(s),m=Qs(d,p,s.shape,u,l,a,i,c,f,h);return e.makeTensorInfo(s.shape,m.dtype,m.values)}const MP={kernelName:cg,backendName:"cpu",kernelFunc:LP};function PP(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reps:r}=s;at(o,"tile");const i=x1(e.bufferSync(o),r);return e.makeTensorInfo(i.shape,i.dtype,i.values)}const zP={kernelName:Wi,backendName:"cpu",kernelFunc:PP};function BP(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{k:r,sorted:i}=s;at(o,"topk");const a=e.data.get(o.dataId).values,[l,c]=y1(a,o.shape,o.dtype,r,i);return[e.makeTensorInfo(l.shape,l.dtype,l.values),e.makeTensorInfo(c.shape,c.dtype,c.values)]}const VP={kernelName:jh,backendName:"cpu",kernelFunc:BP};function WP(n){const{inputs:t,attrs:e,backend:s}=n,{image:o,transforms:r}=t,{interpolation:i,fillMode:a,fillValue:l,outputShape:c}=e,[u,h,d,p]=o.shape,[f,m]=c??[h,d],g=[u,f,m,p],x=ut(o.shape),b=x[0],w=x[1],y=x[2],$=ut(g),v=$[0],N=$[1],T=$[2],k=we(o.dtype,U(g));k.fill(l);const S=s.data.get(o.dataId).values,C=s.data.get(r.dataId).values;for(let O=0;O<u;++O){const P=r.shape[0]===1?C:C.subarray(O*8,O*8+8);for(let M=0;M<f;++M)for(let V=0;V<m;++V)for(let H=0;H<p;++H){let W;const q=P[6]*V+P[7]*M+1;if(q===0)continue;const j=(P[0]*V+P[1]*M+P[2])/q,K=(P[3]*V+P[4]*M+P[5])/q,Y=ym(j,d,a),Z=ym(K,h,a);switch(i){case"nearest":W=KP(S,h,d,b,w,y,O,Z,Y,H,l);break;case"bilinear":W=XP(S,h,d,b,w,y,O,Z,Y,H,l);break;default:throw new Error(`Error in Transform: Expect 'nearest' or 'bilinear', but got ${i}`)}const tt=O*v+M*N+V*T+H;k[tt]=W}return s.makeTensorInfo(g,o.dtype,k)}return{dataId:s.write(k,g,o.dtype),shape:o.shape,dtype:o.dtype}}const UP={kernelName:Kh,backendName:"cpu",kernelFunc:WP};function ym(n,t,e){switch(e){case"reflect":return GP(n,t);case"wrap":return HP(n,t);case"nearest":return jP(n,t);default:return qP(n)}}function GP(n,t){let e=n;if(e<0)if(t<=1)e=0;else{const s=2*t;e<s&&(e=s*Math.trunc(-e/s)+e),e=e<-t?e+s:-e-1}else if(e>t-1)if(t<=1)e=0;else{const s=2*t;e-=s*Math.trunc(e/s),e>=t&&(e=s-e-1)}return Ns(0,e,t-1)}function HP(n,t){let e=n;if(e<0)if(t<=1)e=0;else{const s=t-1;e+=t*(Math.trunc(-e/s)+1)}else if(e>t-1)if(t<=1)e=0;else{const s=t-1;e-=t*Math.trunc(e/s)}return Ns(0,e,t-1)}function qP(n,t){return n}function jP(n,t){return Ns(0,n,t-1)}function vr(n,t,e,s,o,r,i,a,l,c,u){const h=i*s+a*o+l*r+c;return 0<=a&&a<t&&0<=l&&l<e?n[h]:u}function KP(n,t,e,s,o,r,i,a,l,c,u){const h=Math.round(a),d=Math.round(l);return vr(n,t,e,s,o,r,i,h,d,c,u)}function XP(n,t,e,s,o,r,i,a,l,c,u){const h=Math.floor(a),d=Math.floor(l),p=h+1,f=d+1,m=(f-l)*vr(n,t,e,s,o,r,i,h,d,c,u)+(l-d)*vr(n,t,e,s,o,r,i,h,f,c,u),g=(f-l)*vr(n,t,e,s,o,r,i,p,d,c,u)+(l-d)*vr(n,t,e,s,o,r,i,p,f,c,u);return(p-a)*m+(a-h)*g}function YP(n){const{inputs:t,attrs:e,backend:s}=n,{axis:o}=e,{x:r}=t;at(r,"unique");const i=s.data.get(r.dataId).values,{outputValues:a,outputShape:l,indices:c}=w1(i,o,r.shape,r.dtype);return[s.makeTensorInfo(l,r.dtype,a),s.makeTensorInfo([c.length],"int32",c)]}const ZP={kernelName:Xh,backendName:"cpu",kernelFunc:YP};function JP(n){const{inputs:t,backend:e,attrs:s}=n,{value:o}=t;let{axis:r}=s;r<0&&(r+=o.shape.length);const i=o.shape.length,a=o.shape[r],l=new Array(i-1);let c=0;for(let p=0;p<i;p++)p!==r&&(l[c++]=o.shape[p]);const u=new Array(i).fill(0),h=o.shape.slice();h[r]=1;const d=new Array(a);for(let p=0;p<d.length;p++){u[r]=p;const f=wo({inputs:{x:o},backend:e,attrs:{begin:u,size:h}});d[p]=Wt({inputs:{x:f},backend:e,attrs:{shape:l}}),e.disposeIntermediateTensorInfo(f)}return d}const QP={kernelName:fc,backendName:"cpu",kernelFunc:JP};function t3(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,segmentIds:r}=t,{numSegments:i}=s;at(o,"unsortedSegmentSum");const a=o.shape.length,l=r.shape.length,c=[],u=[],h=a-l;let d=r;for(let f=0;f<h;++f){const m=cl({inputs:{input:d},backend:e,attrs:{dim:f+1}});d=m,u.push(m)}for(let f=0;f<i;++f){const m=as(f,"int32"),g=e.makeTensorInfo([],"int32",m),x=Oy({inputs:{a:g,b:d},backend:e}),b=Os({inputs:{x},backend:e,attrs:{dtype:"float32"}}),w=Gc({inputs:{a:b,b:o},backend:e}),y=la({inputs:{x:w},backend:e,attrs:{axis:0,keepDims:!1}});c.push(y),u.push(g),u.push(x),u.push(b),u.push(w),u.push(y)}const p=O1({inputs:c,backend:e,attrs:{axis:0}});return u.forEach(f=>e.disposeIntermediateTensorInfo(f)),p}const e3={kernelName:mc,backendName:"cpu",kernelFunc:t3};const n3=[YF,RD,JF,t_,LD,n_,o_,i_,l_,u_,d_,f_,g_,y_,$_,v_,S_,T_,R_,KF,D_,__,L_,PD,P_,_D,BD,B_,AD,W_,G_,H_,j_,X_,Z_,Q_,eO,sO,rO,aO,cO,hO,pO,mO,gO,bO,wO,CO,IO,vO,kO,NO,RO,VF,DO,VD,BO,WD,VO,GD,jO,KO,YO,qD,KD,JO,tL,nL,oL,YD,JD,DD,iL,U_,lL,uL,dL,WF,tF,nF,fL,oF,gL,yL,$L,vL,SL,TL,EL,iF,AL,FL,OL,ML,zL,VL,UL,lF,HL,KL,ZL,uF,dF,tM,sM,iM,fF,lM,uM,hM,L1,mM,GF,xF,xM,yM,$M,IM,FD,Vu,kM,HF,qF,jF,NM,EM,AM,FM,OM,LM,PM,kF,BM,HM,jM,ZM,NF,QM,eP,sP,TF,XL,iP,lP,uP,dP,fP,gP,bP,wP,AF,$P,FF,OF,IP,kP,NP,EP,AP,zF,TO,FP,OP,MP,zP,VP,UP,mF,ZP,QP,e3,cM];for(const n of n3)Zh(n);const Zs={},ya={alpha:!1,antialias:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,depth:!1,stencil:!1,failIfMajorPerformanceCaveat:!0};function s3(n,t){Zs[n]=t}function Dn(n,t){if(!(n in Zs)||t!=null){const s=r3(n,t);if(s!==null)Zs[n]=s;else return console.log("Could not get context for WebGL version",n),null}const e=Zs[n];return e==null||e.isContextLost()?(delete Zs[n],Dn(n)):(e.disable(e.DEPTH_TEST),e.disable(e.STENCIL_TEST),e.disable(e.BLEND),e.disable(e.DITHER),e.disable(e.POLYGON_OFFSET_FILL),e.disable(e.SAMPLE_COVERAGE),e.enable(e.SCISSOR_TEST),e.enable(e.CULL_FACE),e.cullFace(e.BACK),Zs[n])}function o3(n){if(!B().getBool("IS_SAFARI")&&typeof OffscreenCanvas<"u"&&n===2)return new OffscreenCanvas(300,150);if(typeof document<"u")return document.createElement("canvas");throw new Error("Cannot create a canvas in this context")}function r3(n,t){if(n!==1&&n!==2)throw new Error("Cannot get WebGL rendering context, WebGL is disabled.");const e=t??o3(n);return e.addEventListener("webglcontextlost",s=>{s.preventDefault(),delete Zs[n]},!1),B().getBool("SOFTWARE_WEBGL_ENABLED")&&(ya.failIfMajorPerformanceCaveat=!1),n===1?e.getContext("webgl",ya)||e.getContext("experimental-webgl",ya):e.getContext("webgl2",ya)}var Ur;(function(n){n[n.DENSE=0]="DENSE",n[n.SHARED_BATCH=1]="SHARED_BATCH"})(Ur||(Ur={}));var en;(function(n){n[n.RENDER=0]="RENDER",n[n.UPLOAD=1]="UPLOAD",n[n.PIXELS=2]="PIXELS",n[n.DOWNLOAD=3]="DOWNLOAD"})(en||(en={}));var be;(function(n){n[n.UNPACKED_FLOAT16=0]="UNPACKED_FLOAT16",n[n.UNPACKED_FLOAT32=1]="UNPACKED_FLOAT32",n[n.PACKED_4X1_UNSIGNED_BYTE=2]="PACKED_4X1_UNSIGNED_BYTE",n[n.PACKED_2X2_FLOAT32=3]="PACKED_2X2_FLOAT32",n[n.PACKED_2X2_FLOAT16=4]="PACKED_2X2_FLOAT16"})(be||(be={}));function ca(n,t){return[t,n]}function i3(n,t){return n*t}function wa(n){const t=U(n),e=Math.ceil(t/4);return za(e)}function ir(n,t){return[Math.max(1,Math.ceil(t/2)),Math.max(1,Math.ceil(n/2))]}function a3(n,t){const[e,s]=ir(n,t);return e*s*4}function Gp(n,t){const e=n;let s,o,r,i,a,l,c,u,h,d;return B().getNumber("WEBGL_VERSION")===2?(s=e.R32F,o=e.R16F,r=e.RGBA16F,i=e.RGBA32F,a=e.RED,c=4,u=1,h=e.HALF_FLOAT,d=e.FLOAT,l=e.RGBA8):(s=n.RGBA,o=n.RGBA,r=n.RGBA,i=e.RGBA,a=n.RGBA,c=4,u=4,h=t!=null?t.HALF_FLOAT_OES:null,d=n.FLOAT,l=n.RGBA),{internalFormatFloat:s,internalFormatHalfFloat:o,internalFormatPackedHalfFloat:r,internalFormatPackedFloat:i,textureFormatFloat:a,downloadTextureFormat:l,downloadUnpackNumChannels:c,defaultNumChannels:u,textureTypeHalfFloat:h,textureTypeFloat:d}}function nt(n,t){const e=t();return B().getBool("DEBUG")&&l3(n),e}function l3(n){const t=n.getError();if(t!==n.NO_ERROR)throw new Error("WebGL Error: "+P1(n,t))}const c3=596e-10,u3=65504;function M1(n){return!!(B().getBool("WEBGL_RENDER_FLOAT32_ENABLED")||n===0||c3<Math.abs(n)&&Math.abs(n)<u3)}function P1(n,t){switch(t){case n.NO_ERROR:return"NO_ERROR";case n.INVALID_ENUM:return"INVALID_ENUM";case n.INVALID_VALUE:return"INVALID_VALUE";case n.INVALID_OPERATION:return"INVALID_OPERATION";case n.INVALID_FRAMEBUFFER_OPERATION:return"INVALID_FRAMEBUFFER_OPERATION";case n.OUT_OF_MEMORY:return"OUT_OF_MEMORY";case n.CONTEXT_LOST_WEBGL:return"CONTEXT_LOST_WEBGL";default:return`Unknown error code ${t}`}}function kr(n,t){return fs(n,()=>n.getExtension(t),'Extension "'+t+'" not supported on this browser.')}function z1(n,t){const e=fs(n,()=>n.createShader(n.VERTEX_SHADER),"Unable to create vertex WebGLShader.");if(nt(n,()=>n.shaderSource(e,t)),nt(n,()=>n.compileShader(e)),n.getShaderParameter(e,n.COMPILE_STATUS)===!1)throw console.log(n.getShaderInfoLog(e)),new Error("Failed to compile vertex shader.");return e}function B1(n,t){const e=fs(n,()=>n.createShader(n.FRAGMENT_SHADER),"Unable to create fragment WebGLShader.");if(nt(n,()=>n.shaderSource(e,t)),nt(n,()=>n.compileShader(e)),B().get("ENGINE_COMPILE_ONLY"))return e;if(n.getShaderParameter(e,n.COMPILE_STATUS)===!1)throw Hp(t,n.getShaderInfoLog(e)),new Error("Failed to compile fragment shader.");return e}const h3=/ERROR: [0-9]+:([0-9]+):/g;function Hp(n,t){const e=h3.exec(t);if(e==null){console.log(`Couldn't parse line number in error: ${t}`),console.log(n);return}const s=+e[1],o=n.split(`
`),r=o.length.toString().length+2,i=o.map((h,d)=>to((d+1).toString(),r)+h);let a=0;for(let h=0;h<i.length;h++)a=Math.max(i[h].length,a);const l=i.slice(0,s-1),c=i.slice(s-1,s),u=i.slice(s);console.log(l.join(`
`)),console.log(t.split(`
`)[0]),console.log(`%c ${to(c[0],a)}`,"border:1px solid red; background-color:#e3d2d2; color:#a61717"),console.log(u.join(`
`))}function V1(n){return fs(n,()=>n.createProgram(),"Unable to create WebGLProgram.")}function W1(n,t){if(nt(n,()=>n.linkProgram(t)),!B().get("ENGINE_COMPILE_ONLY")&&n.getProgramParameter(t,n.LINK_STATUS)===!1)throw console.log(n.getProgramInfoLog(t)),new Error("Failed to link vertex and fragment shaders.")}function _a(n,t){if(nt(n,()=>n.validateProgram(t)),n.getProgramParameter(t,n.VALIDATE_STATUS)===!1)throw console.log(n.getProgramInfoLog(t)),new Error("Shader program validation failed.")}function U1(n,t){const e=fs(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return nt(n,()=>n.bindBuffer(n.ARRAY_BUFFER,e)),nt(n,()=>n.bufferData(n.ARRAY_BUFFER,t,n.STATIC_DRAW)),e}function G1(n,t){const e=fs(n,()=>n.createBuffer(),"Unable to create WebGLBuffer");return nt(n,()=>n.bindBuffer(n.ELEMENT_ARRAY_BUFFER,e)),nt(n,()=>n.bufferData(n.ELEMENT_ARRAY_BUFFER,t,n.STATIC_DRAW)),e}function d3(){return B().getNumber("WEBGL_VERSION")===2?1:4}function H1(n){return fs(n,()=>n.createTexture(),"Unable to create WebGLTexture.")}function q1(n,t){const e=B().getNumber("WEBGL_MAX_TEXTURE_SIZE");if(n<=0||t<=0){const s=`[${n}x${t}]`;throw new Error("Requested texture size "+s+" is invalid.")}if(n>e||t>e){const s=`[${n}x${t}]`,o=`[${e}x${e}]`;throw new Error("Requested texture size "+s+" greater than WebGL maximum on this browser / GPU "+o+".")}}function j1(n){return fs(n,()=>n.createFramebuffer(),"Unable to create WebGLFramebuffer.")}function Uu(n,t,e,s,o,r,i){const a=n.getAttribLocation(t,e);return a===-1?!1:(nt(n,()=>n.bindBuffer(n.ARRAY_BUFFER,s)),nt(n,()=>n.vertexAttribPointer(a,o,n.FLOAT,!1,r,i)),nt(n,()=>n.enableVertexAttribArray(a)),!0)}function K1(n,t,e){Q1(n,e),nt(n,()=>n.activeTexture(n.TEXTURE0+e)),nt(n,()=>n.bindTexture(n.TEXTURE_2D,t))}function p3(n,t){Q1(n,t),nt(n,()=>n.activeTexture(n.TEXTURE0+t)),nt(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function X1(n,t,e){return fs(n,()=>n.getUniformLocation(t,e),'uniform "'+e+'" not present in program.')}function Y1(n,t,e){return n.getUniformLocation(t,e)}function Z1(n,t,e,s){nt(n,()=>K1(n,t,s)),nt(n,()=>n.uniform1i(e,s))}function f3(n){nt(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,null)),nt(n,()=>n.viewport(0,0,n.canvas.width,n.canvas.height)),nt(n,()=>n.scissor(0,0,n.canvas.width,n.canvas.height))}function Oa(n,t,e){nt(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,e)),nt(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,t,0))}function Gu(n,t){nt(n,()=>n.bindFramebuffer(n.FRAMEBUFFER,t)),nt(n,()=>n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,null,0))}function Sr(n){const t=n.checkFramebufferStatus(n.FRAMEBUFFER);if(t!==n.FRAMEBUFFER_COMPLETE)throw new Error("Error binding framebuffer: "+J1(n,t))}function J1(n,t){switch(t){case n.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:return"FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";case n.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:return"FRAMEBUFFER_INCOMPLETE_DIMENSIONS";case n.FRAMEBUFFER_UNSUPPORTED:return"FRAMEBUFFER_UNSUPPORTED";default:return`unknown error ${t}`}}function fs(n,t,e){const s=nt(n,()=>t());if(s==null)throw new Error(e);return s}function Q1(n,t){const e=n.MAX_COMBINED_TEXTURE_IMAGE_UNITS-1,s=t+n.TEXTURE0;if(s<n.TEXTURE0||s>e){const o=`[gl.TEXTURE0, gl.TEXTURE${e}]`;throw new Error(`textureUnit must be in ${o}.`)}}function $o(n,t=2){return U(n.slice(0,n.length-t))}function Co(n){if(n.length===0)throw Error("Cannot get rows and columns of an empty shape array.");return[n.length>1?n[n.length-2]:1,n[n.length-1]]}function Nr(n){let t=[1,1,1];return n.length===0||n.length===1&&n[0]===1||(t=[$o(n),...Co(n)]),t}function tw(n,t=!1){let e=B().getNumber("WEBGL_MAX_TEXTURE_SIZE"),s=B().getNumber("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE");s===1/0&&B().getBool("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE")&&(s=e/2),t&&(e=e*2,s=s*2,n=n.map((a,l)=>l>=n.length-2?fl(n[l]):n[l]),n.length===1&&(n=[2,n[0]])),n.length!==2&&(n=is(n).newShape);let o=U(n),r=null;n.length<=1&&o<=e?r=[1,o]:n.length===2&&n[0]<=e&&n[1]<=e?r=n:n.length===3&&n[0]*n[1]<=e&&n[2]<=e?r=[n[0]*n[1],n[2]]:n.length===3&&n[0]<=e&&n[1]*n[2]<=e?r=[n[0],n[1]*n[2]]:n.length===4&&n[0]*n[1]*n[2]<=e&&n[3]<=e?r=[n[0]*n[1]*n[2],n[3]]:n.length===4&&n[0]<=e&&n[1]*n[2]*n[3]<=e&&(r=[n[0],n[1]*n[2]*n[3]]);const i=r!=null&&Math.max(...r)>s&&Math.min(...r)<=(t?2:1)&&Math.min(...r)>0;if(r==null||i)if(t){const a=$o(n);let l=2,c=2;n.length&&([l,c]=Co(n)),o=a*(l/2)*(c/2),r=za(o).map(u=>u*2)}else r=za(o);return r}function $a(n){return n%2===0}function Gr(n,t){if(n=n.slice(-2),t=t.slice(-2),Tt(n,t)||!n.length||!t.length||n[0]===0||n[1]===0||t[0]===0||t[1]===0)return!0;if(n.length!==t.length){const e=n[n.length-1],s=t[t.length-1];if(e===s||$a(e)&&$a(s)&&(n[0]===1||t[0]===1))return!0}return n[1]===t[1]&&$a(n[0])&&$a(t[0])}let La,Ma;function ew(n){if(La==null){const t=Dn(n);La=t.getParameter(t.MAX_TEXTURE_SIZE)}return La}function m3(){La=null}function g3(){Ma=null}function nw(n){if(Ma==null){const t=Dn(n);Ma=t.getParameter(t.MAX_TEXTURE_IMAGE_UNITS)}return Math.min(16,Ma)}function sw(n){if(n===0)return 0;let t;const e=Dn(n);return nn(e,"EXT_disjoint_timer_query_webgl2")&&n===2?t=2:nn(e,"EXT_disjoint_timer_query")?t=1:t=0,t}function nn(n,t){return n.getExtension(t)!=null}function Hu(n){try{if(Dn(n)!=null)return!0}catch(t){return console.log("Error when getting WebGL context: ",t),!1}return!1}function ow(n){if(n===0)return!1;const t=Dn(n);if(n===1){if(!nn(t,"OES_texture_float"))return!1}else if(!nn(t,"EXT_color_buffer_float"))return!1;return qu(t)}function rw(n){if(n===0)return!1;const t=Dn(n);if(n===1){if(!nn(t,"OES_texture_float")||!nn(t,"WEBGL_color_buffer_float"))return!1}else{if(nn(t,"EXT_color_buffer_float"))return qu(t);const s="EXT_color_buffer_half_float";if(nn(t,s)){const o=t.getExtension(s);return x3(t,o)}return!1}return qu(t)}function qu(n){const t=Gp(n),e=n.createTexture();n.bindTexture(n.TEXTURE_2D,e),n.texImage2D(n.TEXTURE_2D,0,t.internalFormatFloat,1,1,0,t.textureFormatFloat,t.textureTypeFloat,null);const r=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,r),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,e,0);const i=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(e),n.deleteFramebuffer(r),i}function x3(n,t){const e=Gp(n,t),s=n.createTexture();n.bindTexture(n.TEXTURE_2D,s),n.texImage2D(n.TEXTURE_2D,0,e.internalFormatHalfFloat,1,1,0,e.textureFormatFloat,e.textureTypeHalfFloat,null);const i=n.createFramebuffer();n.bindFramebuffer(n.FRAMEBUFFER,i),n.framebufferTexture2D(n.FRAMEBUFFER,n.COLOR_ATTACHMENT0,n.TEXTURE_2D,s,0);const a=n.checkFramebufferStatus(n.FRAMEBUFFER)===n.FRAMEBUFFER_COMPLETE;return n.bindTexture(n.TEXTURE_2D,null),n.bindFramebuffer(n.FRAMEBUFFER,null),n.deleteTexture(s),n.deleteFramebuffer(i),a}function iw(n){return n!==2?!1:Dn(n).fenceSync!=null}function ar(n,t){Array.isArray(n)||(n=[n]),n.forEach(e=>{e!=null&&I(e.dtype!=="complex64",()=>`${t} does not support complex64 tensors in the WebGL backend.`)})}const w6=Object.freeze(Object.defineProperty({__proto__:null,assertNotComplex:ar,bindCanvasToFramebuffer:f3,bindColorTextureToFramebuffer:Oa,bindTextureToProgramUniformSampler:Z1,bindTextureUnit:K1,bindVertexBufferToProgramAttribute:Uu,callAndCheck:nt,canBeRepresented:M1,createFragmentShader:B1,createFramebuffer:j1,createProgram:V1,createStaticIndexBuffer:G1,createStaticVertexBuffer:U1,createTexture:H1,createVertexShader:z1,getBatchDim:$o,getExtensionOrThrow:kr,getFramebufferErrorMessage:J1,getMaxTexturesInShader:nw,getNumChannels:d3,getProgramUniformLocation:Y1,getProgramUniformLocationOrThrow:X1,getRowsCols:Co,getShapeAs3D:Nr,getTextureShapeFromLogicalShape:tw,getWebGLDisjointQueryTimerVersion:sw,getWebGLErrorMessage:P1,getWebGLMaxTextureSize:ew,hasExtension:nn,isCapableOfRenderingToFloatTexture:ow,isDownloadFloatTextureEnabled:rw,isReshapeFree:Gr,isWebGLFenceEnabled:iw,isWebGLVersionEnabled:Hu,linkProgram:W1,logShaderSourceAndInfoLog:Hp,resetMaxTextureSize:m3,resetMaxTexturesInShader:g3,unbindColorTextureFromFramebuffer:Gu,unbindTextureUnit:p3,validateFramebuffer:Sr,validateProgram:_a,validateTextureSize:q1},Symbol.toStringTag,{value:"Module"}));const lt=B();lt.registerFlag("HAS_WEBGL",()=>lt.getNumber("WEBGL_VERSION")>0);lt.registerFlag("WEBGL_VERSION",()=>Hu(2)?2:Hu(1)?1:0);lt.registerFlag("WEBGL_CHECK_NUMERICAL_PROBLEMS",()=>!1);lt.registerFlag("WEBGL_BUFFER_SUPPORTED",()=>lt.get("WEBGL_VERSION")===2);lt.registerFlag("WEBGL_CPU_FORWARD",()=>!0);lt.registerFlag("WEBGL_FORCE_F16_TEXTURES",()=>!1);lt.registerFlag("WEBGL_PACK",()=>lt.getBool("HAS_WEBGL"));lt.registerFlag("WEBGL_PACK_NORMALIZATION",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_CLIP",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_DEPTHWISECONV",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_BINARY_OPERATIONS",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_UNARY_OPERATIONS",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_ARRAY_OPERATIONS",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_IMAGE_OPERATIONS",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_REDUCE",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_LAZILY_UNPACK",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_CONV_IM2COL",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_PACK_CONV2DTRANSPOSE",()=>lt.getBool("WEBGL_PACK"));lt.registerFlag("WEBGL_MAX_TEXTURE_SIZE",()=>ew(lt.getNumber("WEBGL_VERSION")));lt.registerFlag("WEBGL_MAX_TEXTURES_IN_SHADER",()=>nw(lt.getNumber("WEBGL_VERSION")));lt.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION",()=>{const n=lt.getNumber("WEBGL_VERSION");return n===0?0:sw(n)});lt.registerFlag("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE",()=>lt.getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0&&!td());lt.registerFlag("WEBGL_RENDER_FLOAT32_CAPABLE",()=>ow(lt.getNumber("WEBGL_VERSION")));lt.registerFlag("WEBGL_RENDER_FLOAT32_ENABLED",()=>lt.getBool("WEBGL_FORCE_F16_TEXTURES")?!1:lt.getBool("WEBGL_RENDER_FLOAT32_CAPABLE"));lt.registerFlag("WEBGL_DOWNLOAD_FLOAT_ENABLED",()=>rw(lt.getNumber("WEBGL_VERSION")));lt.registerFlag("WEBGL_FENCE_API_ENABLED",()=>iw(lt.getNumber("WEBGL_VERSION")));lt.registerFlag("WEBGL_SIZE_UPLOAD_UNIFORM",()=>lt.getBool("WEBGL_RENDER_FLOAT32_ENABLED")?4:0);lt.registerFlag("WEBGL_DELETE_TEXTURE_THRESHOLD",()=>-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_DELETE_TEXTURE_THRESHOLD must be -1 (indicating never delete) or at least 0, but got ${n}.`)});lt.registerFlag("WEBGL_FLUSH_THRESHOLD",()=>td()?1:-1,n=>{if(typeof n!="number")throw new Error(`WEBGL_FLUSH_THRESHOLD must be a number but got ${n}.`);if(n<0&&n!==-1)throw new Error(`WEBGL_FLUSH_THRESHOLD must be -1 (indicating never manual flush) or at least 0, but got ${n}.`)});lt.registerFlag("CPU_HANDOFF_SIZE_THRESHOLD",()=>128);lt.registerFlag("WEBGL_USE_SHAPES_UNIFORMS",()=>!1);lt.registerFlag("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e5);lt.registerFlag("TOPK_K_CPU_HANDOFF_THRESHOLD",()=>128);lt.registerFlag("WEBGL_EXP_CONV",()=>!1);lt.registerFlag("SOFTWARE_WEBGL_ENABLED",()=>lt.getBool("IS_TEST"));lt.registerFlag("WEBGL_MAX_SIZE_FOR_NARROW_TEXTURE",()=>1/0);lt.registerFlag("WEBGL_AUTO_SQUARIFY_NARROW_TEXTURE_SHAPE",()=>!1);lt.registerFlag("WEBGL2_ISNAN_CUSTOM",()=>!1);lt.registerFlag("ENGINE_COMPILE_ONLY",()=>!1);function Le(){let n,t,e,s,o,r,i,a,l,c;return B().getNumber("WEBGL_VERSION")===2?(n="#version 300 es",t="in",e="out",s="in",o="texture",r="outputColor",i="out vec4 outputColor;",a=B().getBool("WEBGL2_ISNAN_CUSTOM")?`
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `:"",l="",c=`
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `):(n="",t="attribute",e="varying",s="varying",o="texture2D",r="gl_FragColor",i="",a=`
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `,l=`
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `,c=`
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `),{version:n,attribute:t,varyingVs:e,varyingFs:s,texture2D:o,output:r,defineOutput:i,defineSpecialNaN:a,defineSpecialInf:l,defineRound:c}}function Ro(n,t,e="index"){const s=ut(t);return s.map((o,r)=>{const i=`int ${n[r]} = ${e} / ${o}`,a=r===s.length-1?`int ${n[r+1]} = ${e} - ${n[r]} * ${o}`:`index -= ${n[r]} * ${o}`;return`${i}; ${a};`}).join("")}function Hc(n,t,e="index"){const s=ut(t);return s.map((o,r)=>{const i=`int ${n[r]} = ${e} / outShapeStrides[${r}]`,a=r===s.length-1?`int ${n[r+1]} = ${e} - ${n[r]} * outShapeStrides[${r}]`:`index -= ${n[r]} * outShapeStrides[${r}]`;return`${i}; ${a};`}).join("")}function b3(n,t){const e=n.length,s=n.map(r=>`${t}[${r}]`),o=new Array(e-1);o[e-2]=s[e-1];for(let r=e-3;r>=0;--r)o[r]=`(${o[r+1]} * ${s[r+1]})`;return o}function y3(n,t,e="index"){const s=n.map((r,i)=>i),o=b3(s,t);return o.map((r,i)=>{const a=`int ${n[i]} = ${e} / ${o[i]}`,l=i===o.length-1?`int ${n[i+1]} = ${e} - ${n[i]} * ${o[i]}`:`index -= ${n[i]} * ${o[i]}`;return`${a}; ${l};`}).join("")}function qp(n){const t=ut(n).map(e=>e.toString());return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * ${t[0]} + coords.y * ${t[1]} + coords.z;
  }
`}function jp(){return`
  int getFlatIndex(ivec3 coords) {
    return coords.x * outShapeStrides[0] + coords.y * outShapeStrides[1] + coords.z;
  }
`}const aw=`
  const float FLOAT_MAX = 1.70141184e38;
  const float FLOAT_MIN = 1.17549435e-38;

  lowp vec4 encode_float(highp float v) {
    if (isnan(v)) {
      return vec4(255, 255, 255, 255);
    }

    highp float av = abs(v);

    if(av < FLOAT_MIN) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    } else if(v > FLOAT_MAX) {
      return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
    } else if(v < -FLOAT_MAX) {
      return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
    }

    highp vec4 c = vec4(0,0,0,0);

    highp float e = floor(log2(av));
    highp float m = exp2(fract(log2(av))) - 1.0;

    c[2] = floor(128.0 * m);
    m -= c[2] / 128.0;
    c[1] = floor(32768.0 * m);
    m -= c[1] / 32768.0;
    c[0] = floor(8388608.0 * m);

    highp float ebias = e + 127.0;
    c[3] = floor(ebias / 2.0);
    ebias -= c[3] * 2.0;
    c[2] += floor(ebias) * 128.0;

    c[3] += 128.0 * step(0.0, -v);

    return c / 255.0;
  }
`;const{getBroadcastDims:lw}=sT;function w3(n,t,e){const s=[];if(n.forEach(p=>{const f=U(p.shapeInfo.logicalShape);if(p.shapeInfo.isUniform?s.push(`uniform float ${p.name}${f>1?`[${f}]`:""};`):(s.push(`uniform sampler2D ${p.name};`),s.push(`uniform int offset${p.name};`)),e.enableShapeUniforms){const{uniformShape:m}=Kp(e.packedInputs,p.shapeInfo.logicalShape,p.shapeInfo.texShape);switch(m.length){case 1:s.push(`uniform int ${p.name}Shape;`);break;case 2:s.push(`uniform ivec2 ${p.name}Shape;`);break;case 3:s.push(`uniform ivec3 ${p.name}Shape;`);break;case 4:s.push(`uniform ivec4 ${p.name}Shape;`);break}s.push(`uniform ivec2 ${p.name}TexShape;`)}}),e.enableShapeUniforms){switch(t.logicalShape.length){case 1:s.push("uniform int outShape;");break;case 2:s.push("uniform ivec2 outShape;"),s.push("uniform int outShapeStrides;");break;case 3:s.push("uniform ivec3 outShape;"),s.push("uniform ivec2 outShapeStrides;");break;case 4:s.push("uniform ivec4 outShape;"),s.push("uniform ivec3 outShapeStrides;");break}s.push("uniform ivec2 outTexShape;")}e.customUniforms&&e.customUniforms.forEach(p=>{s.push(`uniform ${p.type} ${p.name}${p.arrayIndex?`[${p.arrayIndex}]`:""};`)});const o=s.join(`
`),r=n.map(p=>$3(p,t,e.packedInputs,e.enableShapeUniforms)).join(`
`),i=t.texShape,a=Le(),l=v3(a);let c,u,h=N3(a);return t.isPacked?(c=C3(t.logicalShape,i,e.enableShapeUniforms),u=S3(a)):(c=I3(t.logicalShape,i,e.enableShapeUniforms),u=k3(a)),e.packedInputs&&(h+=A3),[h,l,u,o,c,r,e.userCode].join(`
`)}function lr(n,t=!1){const e=n.shapeInfo.logicalShape;switch(e.length){case 0:return U3(n,t);case 1:return H3(n,t);case 2:return j3(n,t);case 3:return X3(n,t);case 4:return Z3(n,t);case 5:return J3(n);case 6:return Q3(n);default:throw new Error(`${e.length}-D input sampling is not yet supported`)}}function cw(n,t){switch(n.shapeInfo.logicalShape.length){case 0:return W3(n);case 1:return G3(n,t);case 2:return q3(n,t);case 3:return K3(n,t);default:return Y3(n,t)}}function $3(n,t,e=!1,s){let o="";e?o+=cw(n,s):o+=lr(n,s);const r=n.shapeInfo.logicalShape,i=t.logicalShape;return r.length<=i.length&&(e?o+=tz(n,t):o+=ez(n,t)),o}function C3(n,t,e){switch(n.length){case 0:return uw();case 1:return D3(n,t,e);case 2:return B3(n,t,e);case 3:return _3(n,t,e);default:return L3(n,t,e)}}function I3(n,t,e){switch(n.length){case 0:return uw();case 1:return F3(n,t,e);case 2:return V3(n,t,e);case 3:return O3(n,t,e);case 4:return M3(n,t,e);case 5:return P3(n,t);case 6:return z3(n,t);default:throw new Error(`${n.length}-D output sampling is not yet supported`)}}function v3(n){return`
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${n.texture2D}(textureSampler, uv).r;
    }
  `}function k3(n){return`
    void setOutput(float val) {
      ${n.output} = vec4(val, 0, 0, 0);
    }
  `}function S3(n){return`
    void setOutput(vec4 val) {
      ${n.output} = val;
    }
  `}function N3(n){return`${n.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${n.varyingFs} vec2 resultUV;
    ${n.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${n.defineSpecialNaN}
    ${n.defineSpecialInf}
    ${n.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${T3}
    ${E3}
    ${R3}
  `}const T3=`
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,E3=`
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,R3=`
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`,A3=`
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;function uw(){return`
    int getOutputCoords() {
      return 0;
    }
  `}function D3(n,t,e){const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];return s[0]===1?e?`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${s[1]}.0);
      }
    `:s[1]===1?e?`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `:`
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${s[0]}.0);
      }
    `:e?`
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      return 2 * (resTexRC.x * ${s[1]} + resTexRC.y);
    }
  `}function F3(n,t,e){return t[0]===1?e?`
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.x * ${t[1]}.0);
      }
    `:t[1]===1?e?`
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `:`
      int getOutputCoords() {
        return int(resultUV.y * ${t[0]}.0);
      }
    `:e?`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `:`
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      return resTexRC.x * ${t[1]} + resTexRC.y;
    }
  `}function _3(n,t,e){if(e)return`
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],o=Math.ceil(n[2]/2),r=o*Math.ceil(n[1]/2);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec3(b, r, c);
    }
  `}function O3(n,t,e){if(e)return`
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${Hc(["r","c","d"],n)}
    return ivec3(r, c, d);
  }
`;const s=Ro(["r","c","d"],n);return`
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${s}
      return ivec3(r, c, d);
    }
  `}function L3(n,t,e){if(e)return`
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)],o=Math.ceil(n[n.length-1]/2),r=o*Math.ceil(n[n.length-2]/2);let i=r,a="",l="b, r, c";for(let c=2;c<n.length-1;c++)i*=n[n.length-c-1],a=`
      int b${c} = index / ${i};
      index -= b${c} * ${i};
    `+a,l=`b${c}, `+l;return`
    ivec${n.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));
      int index = resTexRC.x * ${s[1]} + resTexRC.y;

      ${a}

      int b = index / ${r};
      index -= b * ${r};

      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec${n.length}(${l});
    }
  `}function M3(n,t,e){if(e)return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${Hc(["r","c","d","d2"],n)}
      return ivec4(r, c, d, d2);
    }
  `;const s=Ro(["r","c","d","d2"],n);return`
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      ${s}
      return ivec4(r, c, d, d2);
    }
  `}function P3(n,t){const e=Ro(["r","c","d","d2","d3"],n);return`
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${t[0]},
                             ${t[1]}));

      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${e}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `}function z3(n,t){const e=Ro(["r","c","d","d2","d3","d4"],n);return`
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;

      ${e}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `}function B3(n,t,e){const s=[Math.ceil(t[0]/2),Math.ceil(t[1]/2)];if(Tt(n,t))return e?`
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${s[0]}, ${s[1]}));
      }
    `;const o=Math.ceil(n[1]/2);return e?`
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${s[0]}, ${s[1]}));

      int index = resTexRC.x * ${s[1]} + resTexRC.y;
      int r = 2 * (index / ${o});
      int c = imod(index, ${o}) * 2;

      return ivec2(r, c);
    }
  `}function V3(n,t,e){return Tt(n,t)?e?`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `:`
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${t[0]}, ${t[1]}));
      }
    `:n[1]===1?e?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `:n[0]===1?e?`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `:`
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${t[0]}, ${t[1]}));
        int index = resTexRC.x * ${t[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `:e?`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `:`
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${t[0]}, ${t[1]}));
      int index = resTexRC.x * ${t[1]} + resTexRC.y;
      int r = index / ${n[1]};
      int c = index - r * ${n[1]};
      return ivec2(r, c);
    }
  `}function Ao(n){return`offset${n}`}function W3(n){const t=n.name,e="get"+t.charAt(0).toUpperCase()+t.slice(1),s=Le();return`
    vec4 ${e}() {
      return ${s.texture2D}(${t}, halfCR);
    }
  `}function U3(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1);if(n.shapeInfo.isUniform)return`float ${s}() {return ${e};}`;const[o,r]=n.shapeInfo.texShape;if(o===1&&r===1)return`
      float ${s}() {
        return sampleTexture(${e}, halfCR);
      }
    `;const i=Ao(e);if(t)return`
    float ${s}() {
      vec2 uv = uvFromFlat(${e}TexShape[0], ${e}TexShape[1], ${i});
      return sampleTexture(${e}, uv);
    }
  `;const[a,l]=n.shapeInfo.texShape;return`
    float ${s}() {
      vec2 uv = uvFromFlat(${a}, ${l}, ${i});
      return sampleTexture(${e}, uv);
    }
  `}function G3(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=n.shapeInfo.texShape,r=Le();if(t)return`
    vec4 ${s}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${e}TexShape[0]) / 2.0), ceil(float(${e}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${r.texture2D}(${e}, uv);
    }
  `;const i=[Math.ceil(o[0]/2),Math.ceil(o[1]/2)];return`
    vec4 ${s}(int index) {
      vec2 uv = packedUVfrom1D(
        ${i[0]}, ${i[1]}, index);
      return ${r.texture2D}(${e}, uv);
    }
  `}function H3(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1);if(n.shapeInfo.isUniform)return`
      float ${s}(int index) {
        ${cr(n)}
      }
    `;const o=n.shapeInfo.texShape,r=o[0],i=o[1];if(i===1&&r===1)return`
      float ${s}(int index) {
        return sampleTexture(${e}, halfCR);
      }
    `;const a=Ao(e);return i===1?t?`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / float(${e}TexShape[0]));
        return sampleTexture(${e}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${a}) + 0.5) / ${r}.0);
        return sampleTexture(${e}, uv);
      }
    `:r===1?t?`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / float(${e}TexShape[1]), 0.5);
        return sampleTexture(${e}, uv);
      }
    `:`
      float ${s}(int index) {
        vec2 uv = vec2((float(index + ${a}) + 0.5) / ${i}.0, 0.5);
        return sampleTexture(${e}, uv);
      }
    `:t?`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${e}TexShape[0], ${e}TexShape[1], index + ${a});
      return sampleTexture(${e}, uv);
    }
  `:`
    float ${s}(int index) {
      vec2 uv = uvFromFlat(${r}, ${i}, index + ${a});
      return sampleTexture(${e}, uv);
    }
  `}function q3(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,i=r[0],a=r[1],l=Le();if(r!=null&&Tt(e,r))return t?`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);

        return ${l.texture2D}(${s}, uv);
      }
    `:`
      vec4 ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${a}.0, ${i}.0);

        return ${l.texture2D}(${s}, uv);
      }
    `;if(t)return`
    vec4 ${o}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${l.texture2D}(${s}, uv);
    }
  `;const c=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)],u=Math.ceil(e[1]/2);return`
    vec4 ${o}(int row, int col) {
      vec2 uv = packedUVfrom2D(${u}, ${c[0]}, ${c[1]}, row, col);
      return ${l.texture2D}(${s}, uv);
    }
  `}function j3(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape;if(r!=null&&Tt(e,r)){if(t)return`
      float ${o}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `;const d=r[0],p=r[1];return`
    float ${o}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${p}.0, ${d}.0);
      return sampleTexture(${s}, uv);
    }
  `}const{newShape:i,keptDims:a}=is(e),l=i;if(l.length<e.length){const d=ur(n,l),p=["row","col"];return`
      ${lr(d,t)}
      float ${o}(int row, int col) {
        return ${o}(${hr(p,a)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${e[1]}, 1)));
        ${cr(n)}
      }
    `;const c=r[0],u=r[1],h=Ao(s);return u===1?t?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${h}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${s}TexShape[0]));
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${h}), vec3(${e[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${c}.0);
      return sampleTexture(${s}, uv);
    }
  `:c===1?t?`
      float ${o}(int row, int col) {
        float index = dot(vec3(row, col, ${h}), vec3(${s}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${s}TexShape[1]), 0.5);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col) {
      float index = dot(vec3(row, col, ${h}), vec3(${e[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${u}.0, 0.5);
      return sampleTexture(${s}, uv);
    }
  `:t?`
      float ${o}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${s}Shape[1] + col + ${h};
        vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
        return sampleTexture(${s}, uv);
      }
    `:`
  float ${o}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${e[1]} + col + ${h};
    vec2 uv = uvFromFlat(${c}, ${u}, index);
    return sampleTexture(${s}, uv);
  }
`}function K3(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=n.shapeInfo.texShape,i=[Math.ceil(r[0]/2),Math.ceil(r[1]/2)];if(e[0]===1){const d=e.slice(1),p=[1,2],f=ur(n,d),m=["b","row","col"];return`
        ${cw(f,t)}
        vec4 ${o}(int b, int row, int col) {
          return ${o}(${hr(m,p)});
        }
      `}const a=Le();if(t)return`
    vec4 ${o}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${s}TexShape[0]) / 2.0), ceil(float(${s}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${s}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${s}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `;const l=i[0],c=i[1],u=Math.ceil(e[2]/2),h=u*Math.ceil(e[1]/2);return`
    vec4 ${o}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${l}, ${c}, ${h}, ${u}, b, row, col);
      return ${a.texture2D}(${s}, uv);
    }
  `}function X3(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=e[1]*e[2],i=e[2],{newShape:a,keptDims:l}=is(e),c=a;if(c.length<e.length){const m=ur(n,c),g=["row","col","depth"];return`
        ${lr(m,t)}
        float ${o}(int row, int col, int depth) {
          return ${o}(${hr(g,l)});
        }
      `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${r}, ${i}, 1)));
        ${cr(n)}
      }
    `;const u=n.shapeInfo.texShape,h=u[0],d=u[1],p=n.shapeInfo.flatOffset;if(d===r&&p==null)return t?`
      float ${o}(int row, int col, int depth) {
        int stride1 = ${s}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
        float ${o}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${i}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${d}.0, ${h}.0);
          return sampleTexture(${s}, uv);
        }
      `;if(d===i&&p==null)return t?`
      float ${o}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${s}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
    float ${o}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${e[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${d}.0, ${h}.0);
      return sampleTexture(${s}, uv);
    }
  `;const f=Ao(s);return t?`
    float ${o}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${s}Shape[1] * ${s}Shape[2];
      int stride1 = ${s}Shape[2];
      int index = row * stride0 + col * stride1 + depth + ${f};
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index);
      return sampleTexture(${s}, uv);
    }
    `:`
      float ${o}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${r} + col * ${i} + depth + ${f};
        vec2 uv = uvFromFlat(${h}, ${d}, index);
        return sampleTexture(${s}, uv);
      }
  `}function Y3(n,t){const e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=Le();if(t)return`
    vec4 ${s}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${e}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${e}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${e}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${e}TexShape[0]) / 2.0), ceil(float(${e}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${o.texture2D}(${e}, uv);
    }
  `;const r=n.shapeInfo.logicalShape,i=r.length,a=n.shapeInfo.texShape,l=[Math.ceil(a[0]/2),Math.ceil(a[1]/2)],c=l[0],u=l[1],h=Math.ceil(r[i-1]/2);let d=h*Math.ceil(r[i-2]/2),p="int b, int row, int col",f=`b * ${d} + (row / 2) * ${h} + (col / 2)`;for(let m=2;m<i-1;m++)p=`int b${m}, `+p,d*=r[i-m-1],f=`b${m} * ${d} + `+f;return`
    vec4 ${s}(${p}) {
      int index = ${f};
      int texR = index / ${u};
      int texC = index - texR * ${u};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${u}, ${c});
      return ${o.texture2D}(${e}, uv);
    }
  `}function Z3(n,t){const e=n.shapeInfo.logicalShape,s=n.name,o="get"+s.charAt(0).toUpperCase()+s.slice(1),r=e[3],i=e[2]*r,a=e[1]*i,{newShape:l,keptDims:c}=is(e);if(l.length<e.length){const b=ur(n,l),w=["row","col","depth","depth2"];return`
      ${lr(b,t)}
      float ${o}(int row, int col, int depth, int depth2) {
        return ${o}(${hr(w,c)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${o}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${a}, ${i}, ${r}, 1)));
        ${cr(n)}
      }
    `;const u=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,d=h[0],p=h[1],f=`int stride2 = ${s}Shape[3];`,m=`int stride1 = ${s}Shape[2] * stride2;`,g=`int stride0 = ${s}Shape[1] * stride1;`;if(p===a&&u==null)return t?`
      float ${o}(int row, int col, int depth, int depth2) {
        ${f}
        ${m}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${i}, ${r}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${d}.0);
        return sampleTexture(${s}, uv);
      }
    `;if(p===r&&u==null)return t?`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${s}Shape[1] * ${s}Shape[2], ${s}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${s}TexShape[1], ${s}TexShape[0]);
        return sampleTexture(${s}, uv);
      }
    `:`
      float ${o}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${e[1]*e[2]}, ${e[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${d}.0);
        return sampleTexture(${s}, uv);
      }
    `;const x=Ao(s);return t?`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${f}
      ${m}
      ${g}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${s}TexShape[0], ${s}TexShape[1], index + ${x});
      return sampleTexture(${s}, uv);
    }
  `:`
    float ${o}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} +
          depth * ${r} + depth2;
      vec2 uv = uvFromFlat(${d}, ${p}, index + ${x});
      return sampleTexture(${s}, uv);
    }
  `}function J3(n){const t=n.shapeInfo.logicalShape,e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),o=t[4],r=t[3]*o,i=t[2]*r,a=t[1]*i,{newShape:l,keptDims:c}=is(t);if(l.length<t.length){const m=ur(n,l),g=["row","col","depth","depth2","depth3"];return`
      ${lr(m)}
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        return ${s}(${hr(g,c)});
      }
    `}if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${a}, ${i}, ${r}, ${o})) +
          depth3;
        ${cr(n)}
      }
    `;const u=n.shapeInfo.flatOffset,h=n.shapeInfo.texShape,d=h[0],p=h[1];if(p===a&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${i}, ${r}, ${o}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${p}.0, ${d}.0);
        return sampleTexture(${e}, uv);
      }
    `;if(p===o&&u==null)return`
      float ${s}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]},
               ${t[2]*t[3]}, ${t[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${p}.0, ${d}.0);
        return sampleTexture(${e}, uv);
      }
    `;const f=Ao(e);return`
    float ${s}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${a} + col * ${i} + depth * ${r} +
          depth2 * ${o} + depth3 + ${f};
      vec2 uv = uvFromFlat(${d}, ${p}, index);
      return sampleTexture(${e}, uv);
    }
  `}function Q3(n){const t=n.shapeInfo.logicalShape,e=n.name,s="get"+e.charAt(0).toUpperCase()+e.slice(1),{newShape:o,keptDims:r}=is(t);if(o.length<t.length){const g=ur(n,o),x=["row","col","depth","depth2","depth3","depth4"];return`
      ${lr(g)}
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${s}(${hr(x,r)});
      }
    `}const i=t[5],a=t[4]*i,l=t[3]*a,c=t[2]*l,u=t[1]*c;if(n.shapeInfo.isUniform)return`
      float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${u}, ${c}, ${l}, ${a})) +
          dot(
            vec2(depth3, depth4),
            vec2(${i}, 1)));
        ${cr(n)}
      }
    `;const h=n.shapeInfo.flatOffset,d=n.shapeInfo.texShape,p=d[0],f=d[1];if(f===u&&h==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${c}, ${l}, ${a}, ${i})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${f}.0, ${p}.0);
        return sampleTexture(${e}, uv);
      }
    `;if(f===i&&h==null)return`
      float ${s}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${t[1]*t[2]*t[3]*t[4]},
               ${t[2]*t[3]*t[4]},
               ${t[3]*t[4]},
               ${t[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${f}.0, ${p}.0);
        return sampleTexture(${e}, uv);
      }
    `;const m=Ao(e);return`
    float ${s}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${u} + col * ${c} + depth * ${l} +
          depth2 * ${a} + depth3 * ${i} + depth4 + ${m};
      vec2 uv = uvFromFlat(${p}, ${f}, index);
      return sampleTexture(${e}, uv);
    }
  `}function cr(n){const t=n.name,e=U(n.shapeInfo.logicalShape);return e<2?`return ${t};`:`
    for (int i = 0; i < ${e}; i++) {
      if (i == index) {
        return ${t}[i];
      }
    }
  `}function tz(n,t){const e=n.name,s=e.charAt(0).toUpperCase()+e.slice(1),o="get"+s+"AtOutCoords",r=n.shapeInfo.logicalShape.length,i=t.logicalShape.length,a=lw(n.shapeInfo.logicalShape,t.logicalShape),l=Ot(i),c=i-r;let u;const h=["x","y","z","w","u","v"];r===0?u="":i<2&&a.length>=1?u="coords = 0;":u=a.map(b=>`coords.${h[b+c]} = 0;`).join(`
`);let d="";i<2&&r>0?d="coords":d=n.shapeInfo.logicalShape.map((b,w)=>`coords.${h[w+c]}`).join(", ");let p="return outputValue;";const m=U(n.shapeInfo.logicalShape)===1,x=U(t.logicalShape)===1;if(r===1&&!m&&!x)p=`
      return vec4(outputValue.xy, outputValue.xy);
    `;else if(m&&!x)i===1?p=`
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `:p=`
        return vec4(outputValue.x);
      `;else if(a.length){const b=r-2,w=r-1;a.indexOf(b)>-1&&a.indexOf(w)>-1?p="return vec4(outputValue.x);":a.indexOf(b)>-1?p="return vec4(outputValue.x, outputValue.y, outputValue.x, outputValue.y);":a.indexOf(w)>-1&&(p="return vec4(outputValue.xx, outputValue.zz);")}return`
    vec4 ${o}() {
      ${l} coords = getOutputCoords();
      ${u}
      vec4 outputValue = get${s}(${d});
      ${p}
    }
  `}function ez(n,t){const e=n.name,s=e.charAt(0).toUpperCase()+e.slice(1),o="get"+s+"AtOutCoords",r=t.texShape,i=n.shapeInfo.texShape,a=n.shapeInfo.logicalShape.length,l=t.logicalShape.length;if(!n.shapeInfo.isUniform&&a===l&&n.shapeInfo.flatOffset==null&&Tt(i,r))return`
      float ${o}() {
        return sampleTexture(${e}, resultUV);
      }
    `;const c=Ot(l),u=lw(n.shapeInfo.logicalShape,t.logicalShape),h=l-a;let d;const p=["x","y","z","w","u","v"];a===0?d="":l<2&&u.length>=1?d="coords = 0;":d=u.map(m=>`coords.${p[m+h]} = 0;`).join(`
`);let f="";return l<2&&a>0?f="coords":f=n.shapeInfo.logicalShape.map((m,g)=>`coords.${p[g+h]}`).join(", "),`
    float ${o}() {
      ${c} coords = getOutputCoords();
      ${d}
      return get${s}(${f});
    }
  `}function Ot(n){if(n<=1)return"int";if(n===2)return"ivec2";if(n===3)return"ivec3";if(n===4)return"ivec4";if(n===5)return"ivec5";if(n===6)return"ivec6";throw Error(`GPU for rank ${n} is not yet supported`)}function Kp(n,t,e){const{newShape:s,keptDims:o}=is(t),r=t.length,i=n&&r===3&&t[0]===1,a=i?t.slice(1):s,l=!n&&r>1&&!Tt(t,e)&&s.length<r||i;return{useSqueezeShape:l,uniformShape:l?a:t,keptDims:o}}function ur(n,t){const e=JSON.parse(JSON.stringify(n));return e.shapeInfo.logicalShape=t,e}function hr(n,t){return t.map(e=>n[e]).join(", ")}function nz(n,t,e,s){const o=e.map((u,h)=>{const d={logicalShape:u.shape,texShape:u.isUniform?null:u.texData.texShape,isUniform:u.isUniform,isPacked:u.isUniform?!1:u.texData.isPacked,flatOffset:null};return u.texData!=null&&u.texData.slice!=null&&u.texData.slice.flatOffset>0&&(d.flatOffset=u.texData.slice.flatOffset),{name:t.variableNames[h],shapeInfo:d}}),r=o.map(u=>u.shapeInfo),i={logicalShape:s.shape,texShape:s.texData.texShape,isUniform:!1,isPacked:s.texData.isPacked,flatOffset:null},a=w3(o,i,t),l=B1(n.gl,a),c=n.createProgram(l);return B().get("ENGINE_COMPILE_ONLY")?{program:t,fragmentShader:l,source:a,webGLProgram:c,inShapeInfos:r,outShapeInfo:i,variablesLocations:null,customUniformLocations:null,infLoc:null,nanLoc:null,outShapeLocation:null,outShapeStridesLocation:null,outTexShapeLocation:null}:(n.buildVao(c),Object.assign({program:t,fragmentShader:l,source:a,webGLProgram:c,inShapeInfos:r,outShapeInfo:i},hw(n,t,c)))}function hw(n,t,e){const s=[],o=[];let r,i,a,l=null,c=null;c=n.getUniformLocation(e,"NAN",!1),B().getNumber("WEBGL_VERSION")===1&&(l=n.getUniformLocation(e,"INFINITY",!1));const u=!1;for(const h of t.variableNames){const d={name:h,uniform:n.getUniformLocation(e,h,u),offset:n.getUniformLocation(e,`offset${h}`,u)};t.enableShapeUniforms&&(d.shape=n.getUniformLocation(e,`${h}Shape`,u),d.texShape=n.getUniformLocation(e,`${h}TexShape`,u)),s.push(d)}if(t.enableShapeUniforms&&(r=n.getUniformLocation(e,"outShape",u),a=n.getUniformLocation(e,"outShapeStrides",u),i=n.getUniformLocation(e,"outTexShape",u)),t.customUniforms)for(const h of t.customUniforms)o.push(n.getUniformLocation(e,h.name,u));return{variablesLocations:s,customUniformLocations:o,infLoc:l,nanLoc:c,outShapeLocation:r,outShapeStridesLocation:a,outTexShapeLocation:i}}function wm(n,t){if(n.length!==t.length)throw Error(`Binary was compiled with ${n.length} inputs, but was executed with ${t.length} inputs`);n.forEach((e,s)=>{const o=e.logicalShape,r=t[s],i=r.shape;if(!Tt(o,i))throw Error(`Binary was compiled with different shapes than the current args. Shapes ${o} and ${i} must match`);if(e.isUniform&&r.isUniform)return;const a=e.texShape,l=r.isUniform?null:r.texData.texShape;if(!Tt(a,l))throw Error(`Binary was compiled with different texture shapes than the current args. Shape ${a} and ${l} must match`)})}function sz(n,t,e,s,o){t.program.enableShapeUniforms||(wm(t.inShapeInfos,e),wm([t.outShapeInfo],[s]));const r=s.texData.texture,i=s.texData.texShape;s.texData.isPacked?n.setOutputPackedMatrixTexture(r.texture,i[0],i[1]):n.setOutputMatrixTexture(r.texture,i[0],i[1]),n.setProgram(t.webGLProgram),n.bindVertexArray(t.webGLProgram.vao),B().getNumber("WEBGL_VERSION")===1&&t.infLoc!==null&&n.gl.uniform1f(t.infLoc,1/0),t.nanLoc!==null&&n.gl.uniform1f(t.nanLoc,NaN);for(let l=0;l<e.length;++l){const c=e[l],{uniform:u,offset:h,shape:d,texShape:p}=t.variablesLocations[l];if(d){const{uniformShape:f}=Kp(t.program.packedInputs,c.shape,c.texData.texShape);switch(f.length){case 1:n.gl.uniform1iv(d,new Int32Array(f));break;case 2:n.gl.uniform2iv(d,new Int32Array(f));break;case 3:n.gl.uniform3iv(d,new Int32Array(f));break;case 4:n.gl.uniform4iv(d,new Int32Array(f));break}}if(p&&n.gl.uniform2i(p,c.texData.texShape[0],c.texData.texShape[1]),u!=null){if(c.isUniform){if(U(c.shape)<2)n.gl.uniform1f(u,c.uniformValues[0]);else{let f=c.uniformValues;f instanceof Float32Array||(f=new Float32Array(f)),n.gl.uniform1fv(u,f)}continue}c.texData.slice!=null&&h!=null&&n.gl.uniform1i(h,c.texData.slice.flatOffset),n.setInputMatrixTexture(c.texData.texture.texture,u,l)}}const a=t.outShapeLocation;if(a)switch(s.shape.length){case 1:n.gl.uniform1iv(a,new Int32Array(s.shape));break;case 2:n.gl.uniform2iv(a,new Int32Array(s.shape));break;case 3:n.gl.uniform3iv(a,new Int32Array(s.shape));break;case 4:n.gl.uniform4iv(a,new Int32Array(s.shape));break}if(t.outShapeStridesLocation){const l=ut(s.shape);switch(s.shape.length){case 2:n.gl.uniform1iv(t.outShapeStridesLocation,new Int32Array(l));break;case 3:n.gl.uniform2iv(t.outShapeStridesLocation,new Int32Array(l));break;case 4:n.gl.uniform3iv(t.outShapeStridesLocation,new Int32Array(l));break}}if(t.outTexShapeLocation&&n.gl.uniform2i(t.outTexShapeLocation,s.texData.texShape[0],s.texData.texShape[1]),t.program.customUniforms&&o)for(let l=0;l<t.program.customUniforms.length;++l){const c=t.program.customUniforms[l],u=t.customUniformLocations[l],h=o[l];if(c.type==="float")n.gl.uniform1fv(u,h);else if(c.type==="vec2")n.gl.uniform2fv(u,h);else if(c.type==="vec3")n.gl.uniform3fv(u,h);else if(c.type==="vec4")n.gl.uniform4fv(u,h);else if(c.type==="int")n.gl.uniform1iv(u,h);else if(c.type==="ivec2")n.gl.uniform2iv(u,h);else if(c.type==="ivec3")n.gl.uniform3iv(u,h);else if(c.type==="ivec4")n.gl.uniform4iv(u,h);else throw Error(`uniform type ${c.type} is not supported yet.`)}n.executeProgram()}function oz(n,t,e){let s="";t.concat(e).forEach(i=>{const a=i.texData!=null&&i.texData.slice!=null&&i.texData.slice.flatOffset>0;if(n.enableShapeUniforms&&!i.isUniform){const l=i.texData.texShape,{useSqueezeShape:c,uniformShape:u,keptDims:h}=Kp(n.packedInputs,i.shape,l);let d="",p="",f="";if(u.length===1&&n.packedInputs){const $=[Math.ceil(l[0]/2),Math.ceil(l[1]/2)];d=`${$[0]>1}_${$[1]>1}`}else if(u.length===2&&!n.packedInputs)p=`${u[0]>1}_${u[1]>1}`;else if(u.length>2&&!n.packedInputs){const $=ut(u);f=`${$[0]===l[1]}_${$[$.length-1]===l[1]}`}const m=i.shape.length,g=u.length===2&&Tt(i.shape,l),x=U(i.shape)===1,b=po(i.shape,e.shape),w=!n.packedInputs&&m===e.shape.length&&Tt(l,e.texData.texShape),y=n.packedInputs||u.length>2?"":`${l[0]>1}_${l[1]>1}`;s+=`${m}_${w}_${c?h:""}_${u.length}_${x}_${b}_${g}_${d}_${p}_${f}_${y}_${a}`}else{const l=i.isUniform?"uniform":i.texData.texShape;s+=`${i.shape}_${l}_${a}`}});const o=n.userCode;let r=n.constructor.name;return r+="_"+s+"_"+o+`${B().getNumber("WEBGL_VERSION")}`,r}function Re(n){return B().getBool("WEBGL_USE_SHAPES_UNIFORMS")&&n<=4}class rz{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outPackingScheme=Ur.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const e=Le();this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Hc(["r","c","d"],t):Ro(["r","c","d"],t)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getA(rc.x, rc.y, rc.z);
        }

        ${e.output} = result;
      }
    `}}class iz{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outPackingScheme=Ur.DENSE,this.customUniforms=[{name:"texShape",type:"ivec2"}];const e=Le();this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length),this.userCode=`
      ivec3 outCoordsFromFlatIndex(int index) {
        ${this.enableShapeUniforms?Hc(["r","c","d"],t):Ro(["r","c","d"],t)}
        return ivec3(r, c, d);
      }

      void main() {
        ivec2 resTexRC = ivec2(resultUV.yx * vec2(texShape[0], texShape[1]));
        int index = 4 * (resTexRC.x * texShape[1] + resTexRC.y);

        vec4 result = vec4(0.);

        for (int i=0; i<4; i++) {
          int flatIndex = index + i;
          ivec3 rc = outCoordsFromFlatIndex(flatIndex);
          result[i] = getChannel(getA(rc.x, rc.y, rc.z), vec2(rc.y, rc.z));
        }

        ${e.output} = result;
      }
    `}}class az{constructor(t){this.variableNames=["A"],this.outTexUsage=en.DOWNLOAD;const e=Le();this.outputShape=t,this.userCode=`
      ${aw}

      void main() {
        float x = getAAtOutCoords();
        ${e.output} = encode_float(x);
      }
    `}}class lz{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outTexUsage=en.DOWNLOAD;const e=Le();this.outputShape=t,this.userCode=`
      ${aw}

      void main() {
        ivec3 coords = getOutputCoords();
        float x = getChannel(getAAtOutCoords(), vec2(coords.y, coords.z));
        ${e.output} = encode_float(x);
      }
    `}}const cz={R:0,G:1,B:2,A:3};class $m{constructor(t,e=!1,s="RGBA"){this.variableNames=["A"],this.customUniforms=[{name:"texShape",type:"ivec2"}];const o=Le();this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length);let r="result";e&&(r="floor(result * 255. + 0.5)");let i="";for(let a=0;a<s.length;a++){const l=s[a];i+=`
          if(offset == ${a}) {
            result = values[${cz[l]}];
          }`}this.userCode=`
      ${this.enableShapeUniforms?jp():qp(t)}

      void main() {
        ivec3 coords = getOutputCoords();
        int flatIndex = getFlatIndex(coords);
        float result = 0.;
        int offset = imod(flatIndex, ${s.length});

        flatIndex = idiv(flatIndex, ${s.length}, 1.);

        int r = flatIndex / texShape[1];
        if (r < texShape[0]) {
          int c = imod(flatIndex, texShape[1]);
          vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
          vec4 values = ${o.texture2D}(A, uv);
          ${i}
        }
        ${o.output} = vec4(${r}, 0., 0., 0.);
      }
    `}}class uz{constructor(t,e=!1){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.customUniforms=[{name:"texShape",type:"ivec2"}];const s=Le();this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length);let o="",r="result";e&&(r="floor(result * 255. + 0.5)");for(let i=0;i<=1;i++)for(let a=0;a<=1;a++){const l=i*2+a;o+=`
          localCoords = coords;
          if(localCoords[2] + ${a} < ${this.enableShapeUniforms?"outShape[2]":`${t[2]}`}) {
          localCoords[2] += ${a};
          if (localCoords[1] + ${i} < ${this.enableShapeUniforms?"outShape[1]":`${t[1]}`}) {
            localCoords[1] += ${i};

            flatIndex = getFlatIndex(localCoords);
            offset = imod(flatIndex, 4);

            flatIndex = idiv(flatIndex, 4, 1.);

            int r = flatIndex / texShape[1];
            int c = imod(flatIndex, texShape[1]);
            vec2 uv = (vec2(c, r) + halfCR) / vec2(texShape[1], texShape[0]);
            values = ${s.texture2D}(A, uv);

            if (offset == 0) {
              result[${l}] = values[0];
            } else if (offset == 1) {
              result[${l}] = values[1];
            } else if (offset == 2) {
              result[${l}] = values[2];
            } else {
              result[${l}] = values[3];
            }
          }
        }
        `}this.userCode=`
        ${this.enableShapeUniforms?jp():qp(t)}

        void main() {
          ivec3 coords = getOutputCoords();

          vec4 result = vec4(0.);
          int flatIndex, r, c, offset;
          ivec3 localCoords;
          vec2 uv;
          vec4 values;

          ${o}

          ${s.output} = ${r};
        }
    `}}function dw(n){const t=Le(),e=`${t.version}
    precision highp float;
    ${t.attribute} vec3 clipSpacePos;
    ${t.attribute} vec2 uv;
    ${t.varyingVs} vec2 resultUV;

    void main() {
      gl_Position = vec4(clipSpacePos, 1);
      resultUV = uv;
    }`;return z1(n,e)}function pw(n){const t=new Float32Array([-1,1,0,0,1,-1,-1,0,0,0,1,1,0,1,1,1,-1,0,1,0]);return U1(n,t)}function fw(n){const t=new Uint16Array([0,1,2,2,1,3]);return G1(n,t)}function ua(n,t,e,s,o,r){q1(t,e);const i=H1(n),a=n.TEXTURE_2D;return nt(n,()=>n.bindTexture(a,i)),nt(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_S,n.CLAMP_TO_EDGE)),nt(n,()=>n.texParameteri(a,n.TEXTURE_WRAP_T,n.CLAMP_TO_EDGE)),nt(n,()=>n.texParameteri(a,n.TEXTURE_MIN_FILTER,n.NEAREST)),nt(n,()=>n.texParameteri(a,n.TEXTURE_MAG_FILTER,n.NEAREST)),B().getNumber("WEBGL_VERSION")===1?nt(n,()=>n.texImage2D(a,0,s,t,e,0,o,r,null)):nt(n,()=>n.texStorage2D(a,1,s,t,e)),nt(n,()=>n.bindTexture(n.TEXTURE_2D,null)),{texture:i,texShape:[e,t]}}function Xp(n){return n.internalFormatFloat}function mw(n,t,e,s){const[o,r]=ca(t,e);return ua(n,o,r,Xp(s),s.textureFormatFloat,n.FLOAT)}function Yp(n){return n.internalFormatHalfFloat}function gw(n,t,e,s){const[o,r]=ca(t,e);return ua(n,o,r,Yp(s),s.textureFormatFloat,s.textureTypeHalfFloat)}function Zp(n){return n.downloadTextureFormat}function xw(n,t,e,s){const[o,r]=ca(t,e);return ua(n,o,r,Zp(s),n.RGBA,n.UNSIGNED_BYTE)}function Jp(n){return n.internalFormatPackedFloat}function bw(n,t,e,s){const[o,r]=ir(t,e);return ua(n,o,r,Jp(s),n.RGBA,n.FLOAT)}function Qp(n){return n.internalFormatPackedHalfFloat}function yw(n,t,e,s){const[o,r]=ir(t,e);return ua(n,o,r,Qp(s),n.RGBA,s.textureTypeHalfFloat)}function ww(n,t,e){return nt(n,()=>n.bindBuffer(n.ARRAY_BUFFER,e)),Uu(n,t,"clipSpacePos",e,3,20,0)&&Uu(n,t,"uv",e,2,20,12)}function $w(n,t,e,s,o,r){nt(n,()=>n.bindTexture(n.TEXTURE_2D,t));let i,a,l;o instanceof Uint8Array?(i=new Uint8Array(e*s*4),a=n.UNSIGNED_BYTE,l=n.RGBA):(i=new Float32Array(e*s*4),a=n.FLOAT,l=r.internalFormatPackedFloat),i.set(o),B().getNumber("WEBGL_VERSION")===2?nt(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,e,s,n.RGBA,a,i)):nt(n,()=>n.texImage2D(n.TEXTURE_2D,0,l,e,s,0,n.RGBA,a,i)),nt(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function Cw(n,t,e){nt(n,()=>n.bindTexture(n.TEXTURE_2D,t)),e.data instanceof Uint8Array?B().getNumber("WEBGL_VERSION")===2?nt(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,e.width,e.height,n.RGBA,n.UNSIGNED_BYTE,e.data)):nt(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,e.width,e.height,0,n.RGBA,n.UNSIGNED_BYTE,e.data)):B().getNumber("WEBGL_VERSION")===2?nt(n,()=>n.texSubImage2D(n.TEXTURE_2D,0,0,0,n.RGBA,n.UNSIGNED_BYTE,e)):nt(n,()=>n.texImage2D(n.TEXTURE_2D,0,n.RGBA,n.RGBA,n.UNSIGNED_BYTE,e)),nt(n,()=>n.bindTexture(n.TEXTURE_2D,null))}function Iw(n,t,e,s){const o=n.createBuffer();nt(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,o));const a=4*4*t*e;return nt(n,()=>n.bufferData(n.PIXEL_PACK_BUFFER,a,n.STREAM_READ)),nt(n,()=>n.readPixels(0,0,e,t,n.RGBA,n.FLOAT,0)),nt(n,()=>n.bindBuffer(n.PIXEL_PACK_BUFFER,null)),o}function vw(n,t,e){const s=n,o=new Float32Array(e);return s.bindBuffer(s.PIXEL_PACK_BUFFER,t),s.getBufferSubData(s.PIXEL_PACK_BUFFER,0,o),s.bindBuffer(s.PIXEL_PACK_BUFFER,null),o}function kw(n,t,e,s){const[o,r]=ca(t,e),i=4,a=new Uint8Array(i3(t*e,i));return nt(n,()=>n.readPixels(0,0,o,r,s.downloadTextureFormat,n.UNSIGNED_BYTE,a)),new Float32Array(a.buffer)}function Sw(n,t,e,s,o,r,i,a){const l=n,c=new Float32Array(a3(r,i));return l.bindBuffer(l.PIXEL_PACK_BUFFER,t),l.getBufferSubData(l.PIXEL_PACK_BUFFER,0,c),l.bindBuffer(l.PIXEL_PACK_BUFFER,null),c}function Nw(n,t,e){const s=new Float32Array(t*e*4);return nt(n,()=>n.readPixels(0,0,e,t,n.RGBA,n.FLOAT,s)),s}const $6=Object.freeze(Object.defineProperty({__proto__:null,bindVertexProgramAttributeStreams:ww,createBufferFromOutputTexture:Iw,createFloat16MatrixTexture:gw,createFloat16PackedMatrixTexture:yw,createFloat32MatrixTexture:mw,createIndexBuffer:fw,createPackedMatrixTexture:bw,createUnsignedBytesMatrixTexture:xw,createVertexBuffer:pw,createVertexShader:dw,downloadByteEncodedFloatMatrixFromOutputTexture:kw,downloadFloat32MatrixFromBuffer:vw,downloadMatrixFromPackedOutputTexture:Nw,downloadPackedMatrixFromBuffer:Sw,getInternalFormatForFloat16MatrixTexture:Yp,getInternalFormatForFloat16PackedMatrixTexture:Qp,getInternalFormatForFloat32MatrixTexture:Xp,getInternalFormatForPackedMatrixTexture:Jp,getInternalFormatForUnsignedBytesMatrixTexture:Zp,uploadDenseMatrixToTexture:$w,uploadPixelDataToTexture:Cw},Symbol.toStringTag,{value:"Module"}));class hu{constructor(t){this.outputTexture=null,this.program=null,this.disposed=!1,this.itemsToPoll=[];const e=B().getNumber("WEBGL_VERSION");if(t!=null?(this.gl=t,s3(e,t)):this.gl=Dn(e),t=this.gl,B().getNumber("WEBGL_VERSION")===2){const r=t;this.createVertexArray=()=>nt(r,()=>r.createVertexArray()),this.bindVertexArray=i=>nt(r,()=>r.bindVertexArray(i)),this.deleteVertexArray=i=>nt(r,()=>r.deleteVertexArray(i)),this.getVertexArray=()=>nt(r,()=>r.getParameter(r.VERTEX_ARRAY_BINDING))}else if(t!=null){const r=t.getExtension("OES_vertex_array_object");if(r==null)throw new Error("All WebGL1 implementations are expected to offer OES_vertex_array_object.");this.createVertexArray=()=>nt(t,()=>r.createVertexArrayOES()),this.bindVertexArray=i=>nt(t,()=>r.bindVertexArrayOES(i)),this.deleteVertexArray=i=>nt(t,()=>r.deleteVertexArrayOES(i)),this.getVertexArray=()=>nt(t,()=>t.getParameter(r.VERTEX_ARRAY_BINDING_OES))}let s="WEBGL_color_buffer_float";const o="EXT_color_buffer_half_float";if(this.parallelCompilationExtension=this.gl.getExtension("KHR_parallel_shader_compile"),B().getNumber("WEBGL_VERSION")===1){const r="OES_texture_float",i="OES_texture_half_float";if(this.textureFloatExtension=kr(this.gl,r),nn(this.gl,i))this.textureHalfFloatExtension=kr(this.gl,i);else if(B().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support half float textures, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.");if(this.colorBufferFloatExtension=this.gl.getExtension(s),nn(this.gl,o))this.colorBufferHalfFloatExtension=kr(this.gl,o);else if(B().get("WEBGL_FORCE_F16_TEXTURES"))throw new Error("GL context does not support color renderable half floats, yet the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.")}else if(s="EXT_color_buffer_float",nn(this.gl,s))this.colorBufferFloatExtension=this.gl.getExtension(s);else if(nn(this.gl,o))this.colorBufferHalfFloatExtension=this.gl.getExtension(o);else throw new Error("GL context does not support color renderable floats");this.vertexBuffer=pw(this.gl),this.indexBuffer=fw(this.gl),this.framebuffer=j1(this.gl),this.textureConfig=Gp(this.gl,this.textureHalfFloatExtension)}get debug(){return B().getBool("DEBUG")}dispose(){if(this.disposed)return;this.program!=null&&console.warn("Disposing a GPGPUContext that still has a bound WebGLProgram. This is probably a resource leak, delete the program with GPGPUContext.deleteProgram before disposing."),this.outputTexture!=null&&console.warn("Disposing a GPGPUContext that still has a bound output matrix texture.  This is probably a resource leak, delete the output matrix texture with GPGPUContext.deleteMatrixTexture before disposing.");const t=this.gl;nt(t,()=>t.finish()),nt(t,()=>t.bindFramebuffer(t.FRAMEBUFFER,null)),nt(t,()=>t.deleteFramebuffer(this.framebuffer)),nt(t,()=>t.bindBuffer(t.ARRAY_BUFFER,null)),nt(t,()=>t.bindBuffer(t.ELEMENT_ARRAY_BUFFER,null)),nt(t,()=>t.deleteBuffer(this.indexBuffer)),this.disposed=!0}createFloat32MatrixTexture(t,e){return this.throwIfDisposed(),mw(this.gl,t,e,this.textureConfig)}createFloat16MatrixTexture(t,e){return this.throwIfDisposed(),gw(this.gl,t,e,this.textureConfig)}createUnsignedBytesMatrixTexture(t,e){return this.throwIfDisposed(),xw(this.gl,t,e,this.textureConfig)}uploadPixelDataToTexture(t,e){this.throwIfDisposed(),Cw(this.gl,t,e)}uploadDenseMatrixToTexture(t,e,s,o){this.throwIfDisposed(),$w(this.gl,t,e,s,o,this.textureConfig)}createFloat16PackedMatrixTexture(t,e){return this.throwIfDisposed(),yw(this.gl,t,e,this.textureConfig)}createPackedMatrixTexture(t,e){return this.throwIfDisposed(),bw(this.gl,t,e,this.textureConfig)}deleteMatrixTexture(t){this.throwIfDisposed(),this.outputTexture===t&&(Gu(this.gl,this.framebuffer),this.outputTexture=null),nt(this.gl,()=>this.gl.deleteTexture(t))}downloadByteEncodedFloatMatrixFromOutputTexture(t,e,s){return this.downloadMatrixDriver(t,()=>kw(this.gl,e,s,this.textureConfig))}downloadPackedMatrixFromBuffer(t,e,s,o,r,i){return Sw(this.gl,t,e,s,o,r,i,this.textureConfig)}downloadFloat32MatrixFromBuffer(t,e){return vw(this.gl,t,e)}createBufferFromTexture(t,e,s){this.bindTextureToFrameBuffer(t);const o=Iw(this.gl,e,s,this.textureConfig);return this.unbindTextureToFrameBuffer(),o}createAndWaitForFence(){const t=this.createFence(this.gl);return this.pollFence(t)}createFence(t){let e,s;if(B().getBool("WEBGL_FENCE_API_ENABLED")){const o=t,r=o.fenceSync(o.SYNC_GPU_COMMANDS_COMPLETE,0);t.flush(),s=()=>{const i=o.clientWaitSync(r,0,0);return i===o.ALREADY_SIGNALED||i===o.CONDITION_SATISFIED},e=r}else B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")>0?(e=this.beginQuery(),this.endQuery(),s=()=>this.isQueryAvailable(e,B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))):s=()=>!0;return{query:e,isFencePassed:s}}downloadMatrixFromPackedTexture(t,e,s){return this.downloadMatrixDriver(t,()=>Nw(this.gl,e,s))}createProgram(t){this.throwIfDisposed();const e=this.gl;this.vertexShader==null&&(this.vertexShader=dw(e));const s=V1(e);nt(e,()=>e.attachShader(s,this.vertexShader)),nt(e,()=>e.attachShader(s,t)),W1(e,s);const o=Object.assign(s,{vao:this.createVertexArray()});return this.debug&&_a(e,o),o}buildVao(t){this.setProgram(t),this.bindVertexArray(t.vao);const e=this.gl;nt(e,()=>e.bindBuffer(e.ELEMENT_ARRAY_BUFFER,this.indexBuffer)),ww(e,t,this.vertexBuffer)}deleteProgram(t){this.throwIfDisposed(),t===this.program&&(this.program=null),t!=null&&(nt(this.gl,()=>this.gl.deleteProgram(t)),this.deleteVertexArray(t.vao))}setProgram(t){this.throwIfDisposed(),this.program=t,this.program!=null&&this.debug&&_a(this.gl,this.program),nt(this.gl,()=>this.gl.useProgram(t))}getUniformLocation(t,e,s=!0){return this.throwIfDisposed(),s?X1(this.gl,t,e):Y1(this.gl,t,e)}getAttributeLocation(t,e){return this.throwIfDisposed(),nt(this.gl,()=>this.gl.getAttribLocation(t,e))}getUniformLocationNoThrow(t,e){return this.throwIfDisposed(),this.gl.getUniformLocation(t,e)}setInputMatrixTexture(t,e,s){this.throwIfDisposed(),this.throwIfNoProgram(),Z1(this.gl,t,e,s)}setOutputMatrixTexture(t,e,s){this.setOutputMatrixTextureDriver(t,s,e)}setOutputPackedMatrixTexture(t,e,s){this.throwIfDisposed();const[o,r]=ir(e,s);this.setOutputMatrixTextureDriver(t,o,r)}setOutputMatrixWriteRegion(t,e,s,o){this.setOutputMatrixWriteRegionDriver(s,t,o,e)}setOutputPackedMatrixWriteRegion(t,e,s,o){throw new Error("setOutputPackedMatrixWriteRegion not implemented.")}debugValidate(){this.program!=null&&_a(this.gl,this.program),Sr(this.gl)}executeProgram(){this.throwIfDisposed(),this.throwIfNoProgram();const t=this.gl;if(this.debug){const e=this.getVertexArray();console.assert(e===this.program.vao,"VAO changed between setProgram and executeProgram!"),this.debugValidate()}nt(t,()=>t.drawElements(t.TRIANGLES,6,t.UNSIGNED_SHORT,0))}blockUntilAllProgramsCompleted(){this.throwIfDisposed(),nt(this.gl,()=>this.gl.finish())}getQueryTimerExtension(){return this.disjointQueryTimerExtension==null&&(this.disjointQueryTimerExtension=kr(this.gl,B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2?"EXT_disjoint_timer_query_webgl2":"EXT_disjoint_timer_query")),this.disjointQueryTimerExtension}getQueryTimerExtensionWebGL2(){return this.getQueryTimerExtension()}getQueryTimerExtensionWebGL1(){return this.getQueryTimerExtension()}beginQuery(){if(B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.createQuery();return s.beginQuery(o.TIME_ELAPSED_EXT,r),r}const t=this.getQueryTimerExtensionWebGL1(),e=t.createQueryEXT();return t.beginQueryEXT(t.TIME_ELAPSED_EXT,e),e}endQuery(){if(B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION")===2){const e=this.gl,s=this.getQueryTimerExtensionWebGL2();e.endQuery(s.TIME_ELAPSED_EXT);return}const t=this.getQueryTimerExtensionWebGL1();t.endQueryEXT(t.TIME_ELAPSED_EXT)}async waitForQueryAndGetTime(t){return await pu(()=>this.disposed||this.isQueryAvailable(t,B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))),this.getQueryTime(t,B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION"))}getQueryTime(t,e){if(e===0)return null;if(e===2){const s=this.gl;return s.getQueryParameter(t,s.QUERY_RESULT)/1e6}else{const s=this.getQueryTimerExtensionWebGL1();return s.getQueryObjectEXT(t,s.QUERY_RESULT_EXT)/1e6}}isQueryAvailable(t,e){if(e===0)return!0;if(e===2){const s=this.gl,o=this.getQueryTimerExtensionWebGL2(),r=s.getQueryParameter(t,s.QUERY_RESULT_AVAILABLE);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(o.GPU_DISJOINT_EXT)),r&&!this.disjoint}else{const s=this.getQueryTimerExtensionWebGL1(),o=s.getQueryObjectEXT(t,s.QUERY_RESULT_AVAILABLE_EXT);return this.disjoint==null&&(this.disjoint=this.gl.getParameter(s.GPU_DISJOINT_EXT)),o&&!this.disjoint}}pollFence(t){return new Promise(e=>{this.addItemToPoll(()=>t.isFencePassed(),()=>e())})}pollItems(){const t=hz(this.itemsToPoll.map(e=>e.isDoneFn));for(let e=0;e<=t;++e){const{resolveFn:s}=this.itemsToPoll[e];s()}this.itemsToPoll=this.itemsToPoll.slice(t+1)}addItemToPoll(t,e){if(this.itemsToPoll.push({isDoneFn:t,resolveFn:e}),this.itemsToPoll.length>1)return;let s;"setTimeoutCustom"in B().platform&&(s=B().platform.setTimeoutCustom.bind(B().platform)),pu(()=>(this.pollItems(),this.itemsToPoll.length===0),()=>0,null,s)}bindTextureToFrameBuffer(t){this.throwIfDisposed(),Oa(this.gl,t,this.framebuffer),this.debug&&Sr(this.gl)}unbindTextureToFrameBuffer(){this.outputTexture!=null?(Oa(this.gl,this.outputTexture,this.framebuffer),this.debug&&Sr(this.gl)):Gu(this.gl,this.framebuffer)}downloadMatrixDriver(t,e){this.bindTextureToFrameBuffer(t);const s=e();return this.unbindTextureToFrameBuffer(),s}setOutputMatrixTextureDriver(t,e,s){this.throwIfDisposed();const o=this.gl;Oa(o,t,this.framebuffer),this.debug&&Sr(o),this.outputTexture=t,nt(o,()=>o.viewport(0,0,e,s)),nt(o,()=>o.scissor(0,0,e,s))}setOutputMatrixWriteRegionDriver(t,e,s,o){this.throwIfDisposed(),nt(this.gl,()=>this.gl.scissor(t,e,s,o))}throwIfDisposed(){if(this.disposed)throw new Error("Attempted to use disposed GPGPUContext.")}throwIfNoProgram(){if(this.program==null)throw new Error("No GPU program is currently set.")}}function hz(n){let t=0;for(;t<n.length&&n[t]();++t);return t-1}const{addImpl:dz,bincountImpl:Tw,bincountReduceImpl:pz,bitwiseAndImpl:fz,castImpl:mz,ceilImpl:gz,concatImpl:xz,equalImpl:bz,expImpl:yz,expm1Impl:wz,floorImpl:$z,gatherNdImpl:Cz,gatherV2Impl:Iz,greaterImpl:vz,greaterEqualImpl:kz,lessImpl:Sz,lessEqualImpl:Nz,linSpaceImpl:Tz,logImpl:Ez,maxImpl:Rz,maximumImpl:Az,minimumImpl:Dz,multiplyImpl:Fz,negImpl:_z,notEqualImpl:Oz,prodImpl:Lz,raggedGatherImpl:Mz,raggedRangeImpl:Pz,raggedTensorToTensorImpl:zz,rangeImpl:Bz,rsqrtImpl:Vz,scatterImpl:Wz,sigmoidImpl:Uz,simpleAbsImpl:Ew,sliceImpl:Gz,sparseFillEmptyRowsImpl:Hz,sparseReshapeImpl:qz,sparseSegmentReductionImpl:Rw,sqrtImpl:jz,staticRegexReplaceImpl:Kz,stridedSliceImpl:Xz,stringNGramsImpl:Yz,stringSplitImpl:Zz,stringToHashBucketFastImpl:Jz,subImpl:Qz,tileImpl:tB,topKImpl:eB,transposeImpl:tf,uniqueImpl:nB}=BF;function Aw(n,t){return["x","y","z","w","u","v"].slice(0,t).map(e=>`${n}.${e}`)}function Fe(n,t){return t===1?[n]:Aw(n,t)}function sB(n,t){if(n===1)return"rc";let e="";for(let s=0;s<n;s++)e+=t[s],s<n-1&&(e+=",");return e}class oB{constructor(t){if(this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0,this.outputShape=t,this.rank=t.length,this.enableShapeUniforms=Re(this.outputShape.length),this.rank===0)this.userCode=`
        void main() {
          setOutput(vec4(getA(), 0., 0., 0.));
        }
      `;else{const e=Fe("rc",this.rank),s=Ot(this.rank),o=this.getOutOfBoundsCondition(e),r=this.getSetup(e),i=this.getOutput(e);this.userCode=`
        void main() {
          ${s} rc = getOutputCoords();

          if(${o}) {
            setOutput(vec4(0));
          } else {
            ${r}

            setOutput(vec4(${i}));
          }
        }
      `}}getSourceCoordsArr(t){const e=[];for(let s=0;s<=1;s++)for(let o=0;o<=1;o++){let r=`${s===0?"r":"rp1"}, ${o===0?"c":"cp1"}`;for(let i=2;i<this.rank;i++)r=`${t[t.length-1-i]},`+r;e.push(r)}return e}getOutOfBoundsCondition(t){if(this.rank===1)return`rc > ${this.enableShapeUniforms?"outShape":this.outputShape[0]}`;let e="";for(let s=this.rank-2;s<this.rank;s++)e+=`${t[s]} >= ${this.enableShapeUniforms?`outShape[${s}]`:this.outputShape[s]}`,s<this.rank-1&&(e+="||");return e}getSetup(t){if(this.rank===1)return"";const e=t.slice(-2),s=this.enableShapeUniforms?`outShape[${this.rank} - 1]`:this.outputShape[this.rank-1],o=this.enableShapeUniforms?`outShape[${this.rank} - 2]`:this.outputShape[this.rank-2];return`
      int r = ${e[0]};
      int c = ${e[1]};
      int rp1 = r + 1;
      int cp1 = c + 1;

      bool cEdge = cp1 >= ${s};
      bool rEdge = rp1 >= ${o};
    `}getOutput(t){const e=this.getSourceCoordsArr(t);return this.rank===1?`getA(rc), (rc + 1 >= ${this.enableShapeUniforms?"outShape":this.outputShape[0]} ? 0. : getA(rc + 1)), 0, 0`:`getA(${e[0]}),
            cEdge ? 0. : getA(${e[1]}),
            rEdge ? 0. : getA(${e[2]}),
            rEdge || cEdge ? 0. : getA(${e[3]})`}}class Dw{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec3"}],this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length);let s="";for(let o=0;o<4;o++){let r="thisRC = rc;";o%2===1&&(r+="thisRC.z += 1;"),o>1&&(r+="thisRC.y += 1;"),s+=`
        ${r}
        ${o>0?"if(thisRC.y < rows && thisRC.z < cols){":""}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${o}] =
            getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
        ${o>0?"}":""}
      `}this.userCode=`
      ${rB(e,this.enableShapeUniforms)}
      ${this.enableShapeUniforms?jp():qp(t)}

      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.);

        ivec3 thisRC;
        int rows = ${this.enableShapeUniforms?"outShape[1]":t[1]};
        int cols = ${this.enableShapeUniforms?"outShape[2]":t[2]};

        ${s}

        setOutput(result);
      }
    `}}function rB(n,t){return`
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${t?y3(["r","c","d"],"inputShape"):Ro(["r","c","d"],n)}
      return ivec3(r, c, d);
    }
  `}class iB{constructor(t){this.gpgpu=t,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0,this.freeTextures={},this.usedTextures={},this.logEnabled=!1}acquireTexture(t,e,s){const o=Im(e,s),r=vm(t,o,s);r in this.freeTextures||(this.freeTextures[r]=[]),r in this.usedTextures||(this.usedTextures[r]=[]);const i=Cm(t,o,this.gpgpu.gl,this.gpgpu.textureConfig,s);if(this.freeTextures[r].length>0){this.numFreeTextures--,this.numUsedTextures++,this._numBytesFree-=i,this.log();const l=this.freeTextures[r].pop();return this.usedTextures[r].push(l),l}let a;return o===be.PACKED_2X2_FLOAT32?a=this.gpgpu.createPackedMatrixTexture(t[0],t[1]):o===be.PACKED_2X2_FLOAT16?a=this.gpgpu.createFloat16PackedMatrixTexture(t[0],t[1]):o===be.UNPACKED_FLOAT32?a=this.gpgpu.createFloat32MatrixTexture(t[0],t[1]):o===be.UNPACKED_FLOAT16?a=this.gpgpu.createFloat16MatrixTexture(t[0],t[1]):o===be.PACKED_4X1_UNSIGNED_BYTE&&(a=this.gpgpu.createUnsignedBytesMatrixTexture(t[0],t[1])),this.usedTextures[r].push(a),this.numUsedTextures++,this._numBytesAllocated+=i,this.log(),a}releaseTexture(t,e,s,o){if(this.freeTextures==null)return;const r=Im(s,o),i=vm(e,r,o);i in this.freeTextures||(this.freeTextures[i]=[]);const a=Cm(e,r,this.gpgpu.gl,this.gpgpu.textureConfig,o),l=B().getNumber("WEBGL_DELETE_TEXTURE_THRESHOLD");l!==-1&&this._numBytesAllocated>l?(this.gpgpu.deleteMatrixTexture(t.texture),this._numBytesAllocated-=a):(this.freeTextures[i].push(t),this.numFreeTextures++,this._numBytesFree+=a),this.numUsedTextures--;const c=this.usedTextures[i],u=c&&c.indexOf(t);if(u==null||u<0)throw new Error("Cannot release a texture that was never provided by this texture manager");c[u]=c[c.length-1],c.pop(),this.log()}log(){if(!this.logEnabled)return;const t=this.numFreeTextures+this.numUsedTextures;console.log("Free/Used",`${this.numFreeTextures} / ${this.numUsedTextures}`,`(${t})`);const e=this._numBytesFree/this._numBytesAllocated;console.log(`Bytes allocated: ${this._numBytesAllocated}`),console.log(`Bytes unused: ${this._numBytesFree} (${Math.round(100*e)}%)`)}get numBytesAllocated(){return this._numBytesAllocated}get numBytesFree(){return this._numBytesFree}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){if(this.freeTextures!=null){for(const t in this.freeTextures)this.freeTextures[t].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});for(const t in this.usedTextures)this.usedTextures[t].forEach(e=>{this.gpgpu.deleteMatrixTexture(e.texture)});this.freeTextures=null,this.usedTextures=null,this.numUsedTextures=0,this.numFreeTextures=0,this._numBytesAllocated=0,this._numBytesFree=0}}}function aB(n,t){const e=n;if(t===e.R32F)return 4;if(t===e.R16F)return 2;if(t===e.RGBA32F)return 16;if(t===n.RGBA)return 16;if(t===e.RGBA16F)return 8;if(t===e.RGBA8)return 4;throw new Error(`Unknown internal format ${t}`)}function Cm(n,t,e,s,o){const r=lB(t,s);let i;if(o){const[l,c]=ir(n[0],n[1]);i=l*c}else{const[l,c]=ca(n[0],n[1]);i=l*c}const a=aB(e,r);return i*a}function lB(n,t){switch(n){case be.PACKED_2X2_FLOAT32:return Jp(t);case be.PACKED_2X2_FLOAT16:return Qp(t);case be.UNPACKED_FLOAT32:return Xp(t);case be.UNPACKED_FLOAT16:return Yp(t);case be.PACKED_4X1_UNSIGNED_BYTE:return Zp(t);default:throw new Error(`Unknown physical texture type ${n}`)}}function cB(n){return B().getBool("WEBGL_RENDER_FLOAT32_ENABLED")?n?be.PACKED_2X2_FLOAT32:be.UNPACKED_FLOAT32:n?be.PACKED_2X2_FLOAT16:be.UNPACKED_FLOAT16}function Im(n,t){if(n===en.UPLOAD)return be.PACKED_2X2_FLOAT32;if(n===en.RENDER||n==null)return cB(t);if(n===en.DOWNLOAD||n===en.PIXELS)return be.PACKED_4X1_UNSIGNED_BYTE;throw new Error(`Unknown logical texture type ${n}`)}function vm(n,t,e){return`${n[0]}_${n[1]}_${t}_${e}`}class zn{constructor(t,e){this.variableNames=["A"],this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length),this.userCode=`
      float unaryOperation(float x) {
        ${e}
      }

      void main() {
        float x = getAAtOutCoords();
        float y = unaryOperation(x);

        setOutput(y);
      }
    `}}const yn="if (isnan(x)) return x;",uB="return x;",km="return abs(x);",hB="return (x >= 0.0) ? x : (exp(x) - 1.0);",dB=yn+`
  return (x < 0.0) ? 0.0 : x;
`,pB=yn+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,ys="return x;",fB="return 1.0 / (1.0 + exp(-1.0 * x));";const mB="return x;",gB=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,xB=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,bB=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,yB="return 1.0 / (1.0 + exp(-1.0 * x));";class Is{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length),this.userCode=`
      vec4 unaryOperation(vec4 x) {
        ${e}
      }

      void main() {
        vec4 x = getAAtOutCoords();
        vec4 y = unaryOperation(x);

        setOutput(y);
      }
    `}}class wB{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!1,this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length);const e=t.length,s=Fe("rc",e),o=Ot(e),r=sB(e,s),i=s.slice(-2),a=e<=1?"rc":`vec2(${i.join(",")})`;this.userCode=`
      void main() {
        ${o} rc = getOutputCoords();
        vec4 packedInput = getA(${r});

        setOutput(getChannel(packedInput, ${a}));
      }
    `}}const $B=cx,CB=1e-7,IB=1e-4,Ca={};function vB(n){return n in Ca||(Ca[n]={}),Ca[n]}const kB=B().getNumber("CPU_HANDOFF_SIZE_THRESHOLD"),SB=600;function NB(){return B().global.screen==null?1024:B().global.screen.height*B().global.screen.width*window.devicePixelRatio*SB/1024/1024}class qc extends Yu{nextDataId(){return qc.nextDataId++}constructor(t){if(super(),this.pendingRead=new WeakMap,this.pendingDisposal=new WeakSet,this.dataRefCount=new WeakMap,this.numBytesInGPU=0,this.uploadWaitMs=0,this.downloadWaitMs=0,this.lastGlFlushTime=0,this.warnedAboutMemory=!1,this.pendingDeletes=0,this.disposed=!1,!B().getBool("HAS_WEBGL"))throw new Error("WebGL is not supported on this device");let e;if(t!=null){if(t instanceof hu)e=t;else{const s=Dn(B().getNumber("WEBGL_VERSION"),t);e=new hu(s)}this.binaryCache={},this.gpgpuCreatedLocally=!1}else{const s=Dn(B().getNumber("WEBGL_VERSION"));e=new hu(s),this.binaryCache=vB(B().getNumber("WEBGL_VERSION")),this.gpgpuCreatedLocally=!0}this.gpgpu=e,this.canvas=this.gpgpu.gl.canvas,this.textureManager=new iB(this.gpgpu),this.numMBBeforeWarning=NB(),this.texData=new Gm(this,On())}numDataIds(){return this.texData.numDataIds()-this.pendingDeletes}writeTexture(t,e,s,o,r,i){const a=this.makeTensorInfo(e,s),l=this.texData.get(a.dataId);l.isPacked=!1,l.texture={texture:t,texShape:[o,r]},l.texShape=[o,r];const c=Nr(e),u=new $m(c,!1,i),h=this.runWebGLProgram(u,[a],s,[[o,r]]);return h.shape=e,l.texture=null,this.disposeIntermediateTensorInfo(a),h.dataId}write(t,e,s){if((B().getBool("WEBGL_CHECK_NUMERICAL_PROBLEMS")||B().getBool("DEBUG"))&&this.checkNumericalProblems(t),s==="complex64"&&t!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const o={id:this.nextDataId()};return this.texData.set(o,{shape:e,dtype:s,values:t,usage:en.UPLOAD,refCount:1}),o}refCount(t){return this.texData.has(t)?this.texData.get(t).refCount:0}incRef(t){const e=this.texData.get(t);e.refCount++}decRef(t){if(this.texData.has(t)){const e=this.texData.get(t);e.refCount--}}move(t,e,s,o,r){if(B().getBool("DEBUG")&&this.checkNumericalProblems(e),o==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.texData.set(t,{shape:s,dtype:o,values:e,usage:en.UPLOAD,refCount:r})}disposeIntermediateTensorInfo(t){this.disposeData(t.dataId)}readSync(t){const e=this.texData.get(t),{values:s,dtype:o,complexTensorInfos:r,slice:i,shape:a,isPacked:l}=e;if(i!=null){let d;l?d=new Is(a,ys):d=new zn(a,ys);const p=this.runWebGLProgram(d,[{dataId:t,shape:a,dtype:o}],o),f=this.readSync(p.dataId);return this.disposeIntermediateTensorInfo(p),f}if(s!=null)return this.convertAndCacheOnCPU(t);if(o==="string")return s;const c=this.activeTimers!=null;let u;c&&(u=xe());let h;if(o==="complex64"){const d=this.readSync(r.real.dataId),p=this.readSync(r.imag.dataId);h=os(d,p)}else h=this.getValuesFromTexture(t);return c&&(this.downloadWaitMs+=xe()-u),this.convertAndCacheOnCPU(t,h)}async read(t){if(this.pendingRead.has(t)){const f=this.pendingRead.get(t);return new Promise(m=>f.push(m))}const e=this.texData.get(t),{values:s,shape:o,slice:r,dtype:i,complexTensorInfos:a,isPacked:l}=e;if(r!=null){let f;l?f=new Is(o,ys):f=new zn(o,ys);const m=this.runWebGLProgram(f,[{dataId:t,shape:o,dtype:i}],i),g=this.read(m.dataId);return this.disposeIntermediateTensorInfo(m),g}if(s!=null)return this.convertAndCacheOnCPU(t);if(B().getBool("DEBUG")&&!B().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")&&B().getNumber("WEBGL_VERSION")===2)throw new Error("tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and WEBGL_VERSION=2 not yet supported.");let c=null,u;if(i!=="complex64"&&B().get("WEBGL_BUFFER_SUPPORTED")){u=this.decode(t);const f=this.texData.get(u.dataId);c=this.gpgpu.createBufferFromTexture(f.texture.texture,...wa(o))}this.pendingRead.set(t,[]),i!=="complex64"&&await this.gpgpu.createAndWaitForFence();let h;if(i==="complex64"){const f=await Promise.all([this.read(a.real.dataId),this.read(a.imag.dataId)]),m=f[0],g=f[1];h=os(m,g)}else if(c==null)h=this.getValuesFromTexture(t);else{const f=U(o);h=this.gpgpu.downloadFloat32MatrixFromBuffer(c,f)}if(u!=null&&this.disposeIntermediateTensorInfo(u),c!=null){const f=this.gpgpu.gl;nt(f,()=>f.deleteBuffer(c))}const d=this.convertAndCacheOnCPU(t,h),p=this.pendingRead.get(t);return this.pendingRead.delete(t),p.forEach(f=>f(d)),this.pendingDisposal.has(t)&&(this.pendingDisposal.delete(t),this.disposeData(t)&&On().removeDataId(t,this),this.pendingDeletes--),d}readToGPU(t,e={}){const s=this.texData.get(t),{values:o,shape:r,slice:i,dtype:a,isPacked:l,texture:c}=s;if(a==="complex64")throw new Error("Does not support reading texture for complex64 dtype.");if(i!=null){let p;l?p=new Is(r,ys):p=new zn(r,ys);const f=this.runWebGLProgram(p,[{dataId:t,shape:r,dtype:a}],a),m=this.readToGPU(f,e);return this.disposeIntermediateTensorInfo(f),m}if(c==null)throw o!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const u=this.decode(t,e.customTexShape),h=On().makeTensorFromTensorInfo(u),d=this.texData.get(u.dataId);return Object.assign({tensorRef:h},d.texture)}bufferSync(t){const e=this.readSync(t.dataId);if(t.dtype==="string")try{const s=e.map(o=>ns(o));return Ct(t.shape,t.dtype,s)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return Ct(t.shape,t.dtype,e)}checkNumericalProblems(t){if(t!=null)for(let e=0;e<t.length;e++){const s=t[e];if(!M1(s))throw B().getBool("WEBGL_RENDER_FLOAT32_CAPABLE")?Error(`The value ${s} cannot be represented with your current settings. Consider enabling float32 rendering: 'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`):Error(`The value ${s} cannot be represented on this device.`)}}getValuesFromTexture(t){const{shape:e,dtype:s,isPacked:o}=this.texData.get(t),r=U(e);if(B().getBool("WEBGL_DOWNLOAD_FLOAT_ENABLED")){const d=this.decode(t),p=this.texData.get(d.dataId),f=this.gpgpu.downloadMatrixFromPackedTexture(p.texture.texture,...wa(e)).subarray(0,r);return this.disposeIntermediateTensorInfo(d),f}const i=B().getBool("WEBGL_PACK")&&o===!0,a=i?Nr(e):e,l=i?new lz(a):new az(a),c=this.runWebGLProgram(l,[{shape:a,dtype:s,dataId:t}],"float32"),u=this.texData.get(c.dataId),h=this.gpgpu.downloadByteEncodedFloatMatrixFromOutputTexture(u.texture.texture,u.texShape[0],u.texShape[1]).subarray(0,r);return this.disposeIntermediateTensorInfo(c),h}timerAvailable(){return B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0}time(t){const e=this.activeTimers,s=[];let o=!1;this.programTimersStack==null?(this.programTimersStack=s,o=!0):this.activeTimers.push(s),this.activeTimers=s,t();const r=Es(this.activeTimers.map(l=>l.query)).filter(l=>l!=null),i=Es(this.activeTimers.map(l=>l.name)).filter(l=>l!=null);this.activeTimers=e,o&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null};return(async()=>{if(B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0){const l=await Promise.all(r);a.kernelMs=Hm(l),a.getExtraProfileInfo=()=>l.map((c,u)=>({name:i[u],ms:c})).map(c=>`${c.name}: ${c.ms}`).join(", ")}else a.kernelMs={error:"WebGL query timers are not supported in this environment."};return this.uploadWaitMs=0,this.downloadWaitMs=0,a})()}memory(){return{unreliable:!1,numBytesInGPU:this.numBytesInGPU,numBytesInGPUAllocated:this.textureManager.numBytesAllocated,numBytesInGPUFree:this.textureManager.numBytesFree}}startTimer(){return B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?this.gpgpu.beginQuery():{startMs:xe(),endMs:null}}endTimer(t){return B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0?(this.gpgpu.endQuery(),t):(t.endMs=xe(),t)}async getQueryTime(t){if(B().getNumber("WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE")>0)return this.gpgpu.waitForQueryAndGetTime(t);const e=t;return e.endMs-e.startMs}disposeData(t,e=!1){if(this.pendingDisposal.has(t))return!1;if(!this.texData.has(t))return!0;if(e?this.texData.get(t).refCount=0:this.texData.get(t).refCount--,!e&&this.texData.get(t).refCount>0)return!1;if(this.pendingRead.has(t))return this.pendingDisposal.add(t),this.pendingDeletes++,!1;this.releaseGPUData(t);const{complexTensorInfos:s}=this.texData.get(t);return s!=null&&(this.disposeData(s.real.dataId,e),this.disposeData(s.imag.dataId,e)),this.texData.delete(t),!0}releaseGPUData(t){const{texture:e,dtype:s,texShape:o,usage:r,isPacked:i,slice:a}=this.texData.get(t),l=a&&a.origDataId||t,c=this.dataRefCount.get(l);c>1?this.dataRefCount.set(l,c-1):(this.dataRefCount.delete(l),e!=null&&(this.numBytesInGPU-=this.computeBytes(o,s),this.textureManager.releaseTexture(e,o,r,i)));const u=this.texData.get(t);u.texture=null,u.texShape=null,u.isPacked=!1,u.slice=null}getTexture(t){return this.uploadToGPU(t),this.texData.get(t).texture.texture}getDataInfo(t){return this.texData.get(t)}shouldExecuteOnCPU(t,e=kB){return B().getBool("WEBGL_CPU_FORWARD")&&t.every(s=>this.texData.get(s.dataId).texture==null&&U(s.shape)<e)}getGPGPUContext(){return this.gpgpu}where(t){Qe("tf.where() in webgl locks the UI thread. Call tf.whereAsync() instead");const e=t.dataSync();return $B(t.shape,e)}packedUnaryOp(t,e,s){const o=new Is(t.shape,e),r=this.compileAndRun(o,[t],s);return On().makeTensorFromTensorInfo(r)}abs(t){if(this.shouldExecuteOnCPU([t])&&t.dtype!=="complex64"){const o=Ew(this.texData.get(t.dataId).values);return this.makeOutput(t.shape,t.dtype,o)}if(B().getBool("WEBGL_PACK_UNARY_OPERATIONS"))return this.packedUnaryOp(t,km,t.dtype);const e=new zn(t.shape,km),s=this.compileAndRun(e,[t]);return On().makeTensorFromTensorInfo(s)}makeTensorInfo(t,e,s){let o;if(e==="string"&&s!=null&&s.length>0&&Qo(s[0])){const r=s.map(i=>es(i));o=this.write(r,t,e)}else o=this.write(s,t,e);return this.texData.get(o).usage=null,{dataId:o,shape:t,dtype:e}}makeOutput(t,e,s){return On().makeTensorFromTensorInfo(this.makeTensorInfo(t,e,s),this)}unpackTensor(t){const e=new wB(t.shape);return this.runWebGLProgram(e,[t],t.dtype)}packTensor(t){const e=new oB(t.shape);return this.runWebGLProgram(e,[t],t.dtype,null,!0)}packedReshape(t,e){const s=[$o(t.shape),...Co(t.shape)],o={dtype:t.dtype,shape:s,dataId:t.dataId},r=[$o(e),...Co(e)],i=new Dw(r,s),a=!0,l=[s],c=this.runWebGLProgram(i,[o],t.dtype,l,a);return{dataId:c.dataId,shape:e,dtype:c.dtype}}decode(t,e){const s=this.texData.get(t),{isPacked:o,shape:r,dtype:i}=s;if(e!=null){const d=U(r),p=e[0]*e[1]*4;I(d<=p,()=>"customTexShape is too small. Row * Column * 4 should be equal or larger than the size of the tensor data.")}const a=Nr(r);let l;o?l=new iz(a):l=new rz(a);const c=!0,u=[e??wa(a)],h=this.runWebGLProgram(l,[{shape:a,dtype:i,dataId:t}],i,u,c,e);return{dtype:i,shape:r,dataId:h.dataId}}runWebGLProgram(t,e,s,o,r=!1,i){const a=this.makeTensorInfo(t.outputShape,s),l=this.texData.get(a.dataId);if(t.packedOutput&&(l.isPacked=!0),t.outPackingScheme===Ur.DENSE){const x=i??wa(t.outputShape);l.texShape=x.map(b=>b*2)}if(t.outTexUsage!=null&&(l.usage=t.outTexUsage),U(a.shape)===0)return l.values=we(a.dtype,0),a;const c=[],u=e.map(x=>{if(x.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");let b=this.texData.get(x.dataId);if(b.texture==null){if(!t.packedInputs&&U(x.shape)<=B().getNumber("WEBGL_SIZE_UPLOAD_UNIFORM"))return{shape:x.shape,texData:null,isUniform:!0,uniformValues:b.values};t.packedInputs&&(b.isPacked=!0,b.shape=x.shape)}if(this.uploadToGPU(x.dataId),!!b.isPacked!=!!t.packedInputs)x=b.isPacked?this.unpackTensor(x):this.packTensor(x),c.push(x),b=this.texData.get(x.dataId);else if(b.isPacked&&!Gr(b.shape,x.shape)){const w=x,y=x.shape;x.shape=b.shape,x=this.packedReshape(x,y),c.push(x),b=this.texData.get(x.dataId),w.shape=y}return{shape:x.shape,texData:b,isUniform:!1}});this.uploadToGPU(a.dataId);const h={shape:a.shape,texData:l,isUniform:!1},d=oz(t,u,h),p=this.getAndSaveBinary(d,()=>nz(this.gpgpu,t,u,h)),f=this.activeTimers!=null;let m;f&&(m=this.startTimer()),B().get("ENGINE_COMPILE_ONLY")||sz(this.gpgpu,p,u,h,o),c.forEach(x=>this.disposeIntermediateTensorInfo(x)),f&&(m=this.endTimer(m),this.activeTimers.push({name:t.constructor.name,query:this.getQueryTime(m)}));const g=B().getNumber("WEBGL_FLUSH_THRESHOLD");if(g>0){const x=xe();x-this.lastGlFlushTime>g&&(this.gpgpu.gl.flush(),this.lastGlFlushTime=x)}if(!B().getBool("WEBGL_LAZILY_UNPACK")&&l.isPacked&&r===!1){const x=this.unpackTensor(a);return this.disposeIntermediateTensorInfo(a),x}return a}compileAndRun(t,e,s,o,r=!1){return s=s||e[0].dtype,this.runWebGLProgram(t,e,s,o,r)}getAndSaveBinary(t,e){return t in this.binaryCache||(this.binaryCache[t]=e()),this.binaryCache[t]}getTextureManager(){return this.textureManager}dispose(){this.disposed||(B().getBool("IS_TEST")||Object.keys(this.binaryCache).forEach(e=>{this.gpgpu.deleteProgram(this.binaryCache[e].webGLProgram),delete this.binaryCache[e]}),this.textureManager.dispose(),this.canvas!=null&&typeof HTMLCanvasElement<"u"&&this.canvas instanceof HTMLCanvasElement?this.canvas.remove():this.canvas=null,this.gpgpuCreatedLocally&&(this.gpgpu.program=null,this.gpgpu.dispose()),this.disposed=!0)}floatPrecision(){return this.floatPrecisionValue==null&&(this.floatPrecisionValue=z(()=>{if(!B().get("WEBGL_RENDER_FLOAT32_ENABLED")){const t=B().getBool("DEBUG");B().set("DEBUG",!1);const e=this.abs(mt(1e-8)).dataSync()[0];if(B().set("DEBUG",t),e>0)return 32}return 16})),this.floatPrecisionValue}epsilon(){return this.floatPrecision()===32?CB:IB}uploadToGPU(t){const e=this.texData.get(t),{shape:s,dtype:o,values:r,texture:i,usage:a,isPacked:l}=e;if(i!=null)return;const c=this.activeTimers!=null;let u;c&&(u=xe());let h=e.texShape;if(h==null&&(h=tw(s,l),e.texShape=h),r!=null){const d=Nr(s);let p,f=h[1],m=h[0];const g=r instanceof Uint8Array||r instanceof Uint8ClampedArray;(l||!g)&&([f,m]=ir(h[0],h[1])),l?p=new uz(d,g):p=new $m(d,g);const x=g?[m,f]:h,b=this.makeTensorInfo(x,o),w=this.texData.get(b.dataId);g?w.usage=en.PIXELS:w.usage=en.UPLOAD,w.texShape=x,this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(b.dataId),f,m,r);const y=[[m,f]],v=this.runWebGLProgram(p,[b],o,y,!0),N=this.texData.get(v.dataId);e.texShape=N.texShape,e.isPacked=N.isPacked,e.usage=N.usage,B().get("ENGINE_COMPILE_ONLY")?this.disposeData(v.dataId):(e.texture=N.texture,e.values=null,this.texData.delete(v.dataId)),this.disposeIntermediateTensorInfo(b),c&&(this.uploadWaitMs+=xe()-u)}else{const d=this.acquireTexture(h,a,o,l);e.texture=d}}convertAndCacheOnCPU(t,e){const s=this.texData.get(t),{dtype:o}=s;return e!=null&&(s.values=TB(e,o)),s.values}acquireTexture(t,e,s,o){if(this.numBytesInGPU+=this.computeBytes(t,s),!this.warnedAboutMemory&&this.numBytesInGPU>this.numMBBeforeWarning*1024*1024){const r=(this.numBytesInGPU/1024/1024).toFixed(2);this.warnedAboutMemory=!0,console.warn(`High memory usage in GPU: ${r} MB, most likely due to a memory leak`)}return this.textureManager.acquireTexture(t,e,o)}computeBytes(t,e){return t[0]*t[1]*Ar(e)}checkCompileCompletion(){for(const[,t]of Object.entries(this.binaryCache))this.checkCompletion_(t)}async checkCompileCompletionAsync(){const t=[];if(this.gpgpu.parallelCompilationExtension){for(const[,e]of Object.entries(this.binaryCache))t.push(this.checkCompletionAsync_(e));return Promise.all(t)}else{for(const[,e]of Object.entries(this.binaryCache)){const s=new Promise(o=>{try{this.checkCompletion_(e),o(!0)}catch(r){throw r}});t.push(s)}return Promise.all(t)}}async checkCompletionAsync_(t){return this.gpgpu.gl.getProgramParameter(t.webGLProgram,this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)?this.checkCompletion_(t):(await Dx(),this.checkCompletionAsync_(t))}checkCompletion_(t){if(this.gpgpu.gl.getProgramParameter(t.webGLProgram,this.gpgpu.gl.LINK_STATUS)===!1)throw console.log(this.gpgpu.gl.getProgramInfoLog(t.webGLProgram)),this.gpgpu.gl.getShaderParameter(t.fragmentShader,this.gpgpu.gl.COMPILE_STATUS)===!1?(Hp(t.source,this.gpgpu.gl.getShaderInfoLog(t.fragmentShader)),new Error("Failed to compile fragment shader.")):new Error("Failed to link vertex and fragment shaders.");return!0}getUniformLocations(){for(const t of Object.values(this.binaryCache)){this.gpgpu.buildVao(t.webGLProgram);const{variablesLocations:e,customUniformLocations:s,infLoc:o,nanLoc:r,outShapeLocation:i,outShapeStridesLocation:a,outTexShapeLocation:l}=hw(this.gpgpu,t.program,t.webGLProgram);t.variablesLocations=e,t.customUniformLocations=s,t.infLoc=o,t.nanLoc=r,t.outShapeLocation=i,t.outShapeStridesLocation=a,t.outTexShapeLocation=l}}createTensorFromGPUData(t,e,s){t.channels=t.channels||"RGBA";const{texture:o,height:r,width:i,channels:a}=t,l=On().backend;if(!l.gpgpu.gl.isTexture(o))throw new Error("The texture is invalid. Also, please make sure the texture and the TFJS WebGL backend are using the same canvas. If you want to use your own custom canvas, you have to create and use the custom TFJS WebGL backend created from the canvas through 'new tf.MathBackendWebGL(customCanvas)'.");const c=l.writeTexture(o,e,s,r,i,a);return On().makeTensorFromDataId(c,e,s,l)}}qc.nextDataId=0;function TB(n,t){if(t==="float32"||t==="complex64")return n;if(t==="int32"||t==="bool"){const e=t==="int32"?new Int32Array(n.length):new Uint8Array(n.length);for(let s=0;s<e.length;++s)e[s]=Math.round(n[s]);return e}else throw new Error(`Unknown dtype ${t}`)}function EB(){B().set("WEBGL_FORCE_F16_TEXTURES",!0)}ed()&&kg("webgl",()=>new qc,2);const C6={forceHalfFloat:EB};const ef=`
  if (isnan(a)) return a;
  if (isnan(b)) return b;
`;class Io{constructor(t,e,s){this.variableNames=["A","B"],this.outputShape=gt(e,s),this.enableShapeUniforms=Re(this.outputShape.length),this.userCode=`
      float binaryOperation(float a, float b) {
        ${t}
      }

      void main() {
        float a = getAAtOutCoords();
        float b = getBAtOutCoords();
        setOutput(binaryOperation(a, b));
      }
    `}}const Do=`
  result.r = isNaN.r ? NAN : result.r;
  result.g = isNaN.g ? NAN : result.g;
  result.b = isNaN.b ? NAN : result.b;
  result.a = isNaN.a ? NAN : result.a;
`;class dr{constructor(t,e,s,o=!1){this.variableNames=["A","B"],this.supportsBroadcasting=!0,this.packedInputs=!0,this.packedOutput=!0,this.outputShape=gt(e,s);const r=this.outputShape.length;this.enableShapeUniforms=Re(r);let i="";if(o)if(r===0||U(this.outputShape)===1)i=`
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;else if(i=`
          ${Ot(r)} coords = getOutputCoords();
        `,r===1)this.enableShapeUniforms?i+=`
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `:i+=`
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;else{const l=Fe("coords",r);this.enableShapeUniforms?i+=`
            bool nextRowOutOfBounds =
              (${l[r-2]} + 1) >= outShape[${r} - 2];
            bool nextColOutOfBounds =
              (${l[r-1]} + 1) >= outShape[${r} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `:i+=`
            bool nextRowOutOfBounds =
              (${l[r-2]} + 1) >= ${this.outputShape[r-2]};
            bool nextColOutOfBounds =
              (${l[r-1]} + 1) >= ${this.outputShape[r-1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `}this.userCode=`
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${t}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${i}

        setOutput(result);
      }
    `}}function Ye(n){const{inputs:t,backend:e}=n,{x:s}=t;return e.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const RB={kernelName:mi,backendName:"webgl",kernelFunc:Ye};function Gs(n){const{inputs:t,backend:e}=n,{real:s,imag:o}=t,r=e.makeTensorInfo(s.shape,"complex64"),i=e.texData.get(r.dataId),a=Ye({inputs:{x:s},backend:e}),l=Ye({inputs:{x:o},backend:e});return i.complexTensorInfos={real:a,imag:l},r}const AB={kernelName:uh,backendName:"webgl",kernelFunc:Gs};const Fw="return (a < 0.) ? b * a : a;",_w=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function DB(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{alpha:r}=s,i=e.makeTensorInfo([],"float32",as(r,"float32")),a=B().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new dr(_w,o.shape,i.shape):new Io(Fw,o.shape,i.shape),l=e.runWebGLProgram(a,[o,i],"float32");return e.disposeIntermediateTensorInfo(i),l}const FB={kernelName:Ll,backendName:"webgl",kernelFunc:DB};const Ow="return (a < 0.) ? b * a : a;",Lw=`
  vec4 aLessThanZero = vec4(lessThan(a, vec4(0.)));
  return (aLessThanZero * (b * a)) + ((vec4(1.0) - aLessThanZero) * a);
`;function _B(n){const{inputs:t,backend:e}=n,{x:s,alpha:o}=t,r=B().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new dr(Lw,s.shape,o.shape):new Io(Ow,s.shape,o.shape);return e.runWebGLProgram(r,[s,o],"float32")}const OB={kernelName:ec,backendName:"webgl",kernelFunc:_B};const pr="if (isnan(x)) return x;";function St({opSnippet:n,packedOpSnippet:t,cpuKernelImpl:e,dtype:s}){return({inputs:o,backend:r})=>{const{x:i}=o,a=r,l=s||i.dtype;if(a.shouldExecuteOnCPU([i])&&e!=null){const h=a.texData.get(i.dataId),d=e(h.values,l);return a.makeTensorInfo(i.shape,l,d)}const c=B().getBool("WEBGL_PACK_UNARY_OPERATIONS")&&t!=null;let u;return c?u=new Is(i.shape,t):u=new zn(i.shape,n),a.runWebGLProgram(u,[i],l)}}function ve({opSnippet:n,packedOpSnippet:t,checkOutOfBounds:e=!1,supportsComplex:s=!1,cpuKernelImpl:o,dtype:r}){return({inputs:i,backend:a})=>{const{a:l,b:c}=i,u=a;if(s&&l.dtype==="complex64"){const f=u.texData.get(l.dataId),m=u.texData.get(c.dataId),[g,x]=[[f.complexTensorInfos.real,m.complexTensorInfos.real],[f.complexTensorInfos.imag,m.complexTensorInfos.imag]].map(w=>{const[y,$]=w,v={dataId:y.dataId,dtype:y.dtype,shape:l.shape},N={dataId:$.dataId,dtype:$.dtype,shape:c.shape},T=new Io(n,l.shape,c.shape);return u.runWebGLProgram(T,[v,N],Ke(y.dtype,$.dtype))}),b=Gs({inputs:{real:g,imag:x},backend:u});return u.disposeIntermediateTensorInfo(g),u.disposeIntermediateTensorInfo(x),b}const h=r||Ke(l.dtype,c.dtype);if((l.dtype==="string"||c.dtype==="string"||u.shouldExecuteOnCPU([l,c]))&&o!=null){const f=u.texData.get(l.dataId).values,m=u.texData.get(c.dataId).values,g=l.dtype==="string"?rs(f):f,x=l.dtype==="string"?rs(m):m,[b,w]=o(l.shape,c.shape,g,x,h),y=u.makeTensorInfo(w,h),$=u.texData.get(y.dataId);return $.values=b,y}const d=B().getBool("WEBGL_PACK_BINARY_OPERATIONS")&&t!=null;let p;return d?p=new dr(t,l.shape,c.shape,e):p=new Io(n,l.shape,c.shape),u.runWebGLProgram(p,[l,c],h)}}function Hr(n,t=!1){if(n==="linear")return t?mB:uB;if(n==="relu")return t?xB:dB;if(n==="elu")return t?gB:hB;if(n==="relu6")return t?bB:pB;if(n==="prelu")return t?Lw:Ow;if(n==="leakyrelu")return t?_w:Fw;if(n==="sigmoid")return t?yB:fB;throw new Error(`Activation ${n} has not been implemented for the WebGL backend.`)}class Mw{constructor(t,e,s,o=!1,r=!1,i=!1,a=null,l=!1,c=!1){this.variableNames=["matrixA","matrixB"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=s,this.enableShapeUniforms=Re(this.outputShape.length);const u=o?t[1]:t[2],h=Math.ceil(u/2),d=o?"i * 2, rc.y":"rc.y, i * 2",p=r?"rc.z, i * 2":"i * 2, rc.z",f=o?["a.xxyy","a.zzww"]:["a.xxzz","a.yyww"],m=r?["b.xzxz","b.ywyw"]:["b.xyxy","b.zwzw"];let g="",x="";a&&(l?g=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${a}
        }`:c?g=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${a}
        }`:g=`vec4 activation(vec4 x) {
          ${a}
        }`,x="result = activation(result);");const b=i?"result += getBiasAtOutCoords();":"";i&&this.variableNames.push("bias"),l&&this.variableNames.push("preluActivationWeights"),c&&this.variableNames.push("leakyreluAlpha");let w="rc.x",y="rc.x";t[0]<e[0]?w=`imod(rc.x, ${t[0]})`:e[0]<t[0]&&(y=`imod(rc.x, ${e[0]})`),this.userCode=`
      ${g}
      // Don't use uniform for sharedDimensionPacked for performance.
      const float sharedDimension = ${h}.0;

      vec4 dot2x2ARowBCol(ivec3 rc) {
        vec4 result = vec4(0);
        int batchA = ${w};
        int batchB = ${y};
        for (int i = 0; i < ${h}; i++) {
          vec4 a = getMatrixA(batchA, ${d});
          vec4 b = getMatrixB(batchB, ${p});

          // These swizzled products need to be separately added.
          // See: https://github.com/tensorflow/tfjs/issues/1735
          result += (${f[0]} * ${m[0]});
          result += (${f[1]} * ${m[1]});
        }
        return result;
      }

      void main() {
        ivec3 rc = getOutputCoords();
        vec4 result = dot2x2ARowBCol(rc);

        ${b}

        ${x}

        setOutput(result);
      }
    `}}const Sm={REAL:"return areal * breal - aimag * bimag;",IMAG:"return areal * bimag + aimag * breal;"};class Nm{constructor(t,e,s){this.variableNames=["AReal","AImag","BReal","BImag"],this.outputShape=gt(e,s),this.userCode=`
      float binaryOpComplex(
          float areal, float aimag, float breal, float bimag) {
        ${t}
      }

      void main() {
        float areal = getARealAtOutCoords();
        float aimag = getAImagAtOutCoords();
        float breal = getBRealAtOutCoords();
        float bimag = getBImagAtOutCoords();
        setOutput(binaryOpComplex(areal, aimag, breal, bimag));
      }
    `}}const Tm="return a * b;";function nf(n){const{inputs:t,backend:e}=n,{a:s,b:o}=t,r=Ke(s.dtype,o.dtype);if(s.dtype==="complex64"){const a=e.texData.get(s.dataId),l=e.texData.get(o.dataId),c=new Nm(Sm.REAL,s.shape,o.shape),u=new Nm(Sm.IMAG,s.shape,o.shape),h=[{dataId:a.complexTensorInfos.real.dataId,dtype:a.complexTensorInfos.real.dtype,shape:s.shape},{dataId:a.complexTensorInfos.imag.dataId,dtype:a.complexTensorInfos.imag.dtype,shape:s.shape},{dataId:l.complexTensorInfos.real.dataId,dtype:l.complexTensorInfos.real.dtype,shape:o.shape},{dataId:l.complexTensorInfos.imag.dataId,dtype:l.complexTensorInfos.imag.dtype,shape:o.shape}],d=e.runWebGLProgram(c,h,"float32"),p=e.runWebGLProgram(u,h,"float32"),f=Gs({inputs:{real:d,imag:p},backend:e});return e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),f}if(e.shouldExecuteOnCPU([s,o])){const a=e.texData.get(s.dataId),l=e.texData.get(o.dataId),[c,u]=Fz(s.shape,o.shape,a.values,l.values,r),h=e.makeTensorInfo(u,r),d=e.texData.get(h.dataId);return d.values=c,h}let i;return B().getBool("WEBGL_PACK_BINARY_OPERATIONS")?i=new dr(Tm,s.shape,o.shape):i=new Io(Tm,s.shape,o.shape),e.runWebGLProgram(i,[s,o],r)}const LB={kernelName:vi,backendName:"webgl",kernelFunc:nf};function MB(n,t,e){const s=[$o(n.shape),...Co(n.shape)],o={dtype:n.dtype,shape:s,dataId:n.dataId},r=[$o(t),...Co(t)],i=new Dw(r,s),a=!0,l=[s],c=e.runWebGLProgram(i,[o],n.dtype,l,a);return{dataId:c.dataId,shape:t,dtype:c.dtype}}function et(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{shape:r}=s,i=e,a=U(o.shape),l=Qu(r,a),c=U(l);I(a===c,()=>`The new shape (${l}) has ${c} elements and the old shape (${o.shape}) has ${a} elements. The new shape and old shape must have the same number of elements.`);const u=i.texData.get(o.dataId);return u.isPacked&&!Gr(o.shape,l)&&!(u.texture!==null&&Gr(u.shape,l))?MB(o,l,i):(i.incRef(o.dataId),{dataId:o.dataId,shape:l,dtype:o.dtype})}const PB={kernelName:sc,backendName:"webgl",kernelFunc:et};class Em{constructor(t,e){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:i}=t;this.outputShape=[o,i];const a=Math.floor(s/4)*4,l=s%4;let c="sumValue += dot(values, ones);";if(e!=null){const h=1/e;c=`sumValue += dot(values * ${oo(h)?h.toPrecision(2):h}, ones);`}let u="";r%s>0&&(u=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return 0.0;
        }
      `),this.userCode=`
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${u}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        float sumValue = 0.0;

        for (int i = 0; i < ${a}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${c}
        }

        int inIdx = inOffset + ${a};
        if (${l===1}) {
          vec4 values = vec4(getValue(batch, inIdx), 0.0, 0.0, 0.0);

          ${c}
        } else if (${l===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1), 0.0, 0.0);

          ${c}
        } else if (${l===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2), 0.0);

          ${c}
        }
        setOutput(sumValue);
      }
    `}}class zB{constructor(t,e){this.variableNames=["x"];const{windowSize:s,batchSize:o,inSize:r,outSize:i}=t;this.outputShape=[o,i];let a="0.0",l="";e==="prod"?a="1.0":e==="min"?(a="1.0 / 1e-20",l="min"):e==="max"&&(a="-1.0 / 1e-20",l="max");let c=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="sum"?c="sumValue":e==="prod"?c="prodValue":e==="all"?c="allValue":e==="any"&&(c="anyValue");const u=Math.floor(s/4)*4,h=s%4;let d=`
      if (${e==="sum"}) {
        sumValue += dot(values, ones);
      } else if (${e==="prod"}) {
        vec2 tmp = vec2(values[0], values[1]) * vec2(values[2], values[3]);
        prodValue *= tmp[0] * tmp[1];
      } else {
        minMaxValue = ${l}(values, minMaxValue);
        if (${e==="min"} || ${e==="max"}) {
          minMaxValue = ${l}(values, minMaxValue);
          bvec4 isNaN = isnan(values);
          if (isNaN.r || isNaN.g || isNaN.b || isNaN.a) {
            minMaxValue = vec4(NAN);
          }
        }
      }
    `,p="vec4";e==="all"?(a="1.0",d=`
        bool reducedAllValue = all(values);
        float floatedReducedAllValue = float(reducedAllValue);
        allValue = float(allValue >= 1.0 && floatedReducedAllValue >= 1.0);
      `,p="bvec4"):e==="any"&&(a="0.0",d=`
        bool reducedAnyValue = any(values);
        float floatedReducedAnyValue = float(reducedAnyValue);
        anyValue = float(anyValue >= 1.0 || floatedReducedAnyValue >= 1.0);
      `,p="bvec4");let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `),this.userCode=`
      const float initializationValue = ${a};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float getValue(int batch, int inIdx) {
        ${f}
        return getX(batch, inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${s};

        vec4 minMaxValue = vec4(${a});
        float prodValue = 1.0;
        float sumValue = 0.0;
        float allValue = 1.0;
        float anyValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          ${d}
        }

        int inIdx = inOffset + ${u};
        if (${h===1}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${h===2}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          ${d}
        } else if (${h===3}) {
          ${p} values = ${p}(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          ${d}
        }
        setOutput(${c});
      }
    `}}function BB(n){const t=[];for(;t.length===0||t[t.length-1].outSize!==1;){const e=t.length?t[t.length-1].outSize:n[1],s=Tc(e);t.push({inSize:e,windowSize:s,outSize:Math.ceil(e/s)})}return t}function Fo(n,t,e,s){const o=BB(n.shape);let r=n;for(let i=0;i<o.length;i++){const{inSize:a,windowSize:l,outSize:c}=o[i];let u,h;e==="mean"?u=i===0?new Em({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c},a):new Em({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c}):u=new zB({windowSize:l,inSize:a,batchSize:n.shape[0],outSize:c},e),h=r,r=s.runWebGLProgram(u,[r],t),h.dataId!==n.dataId&&s.disposeIntermediateTensorInfo(h)}return r}class VB{constructor(t,e){this.variableNames=["A"];const s=new Array(t.length);for(let i=0;i<s.length;i++)s[i]=t[e[i]];this.outputShape=s,this.rank=s.length;const o=Ot(this.rank),r=WB(e);this.userCode=`
    void main() {
      ${o} resRC = getOutputCoords();
      setOutput(getA(${r}));
    }
    `}}function WB(n){const t=n.length;if(t>6)throw Error(`Transpose for rank ${t} is not yet supported`);const e=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u","resRC.v"],s=new Array(t);for(let o=0;o<n.length;o++)s[n[o]]=e[o];return s.join()}class UB{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0;const s=new Array(t.length);for(let u=0;u<s.length;u++)s[u]=t[e[u]];if(this.outputShape=s,this.rank=s.length,this.rank>6)throw Error(`Packed transpose for rank ${this.rank} is not yet supported.`);const o=Ot(this.rank),r=Aw("rc",this.rank),i=new Array(this.rank);for(let u=0;u<e.length;u++)i[e[u]]=r[u];const a=`vec2(${i.slice(-2).join()})`,l=`++${r[this.rank-1]} < ${s[this.rank-1]}`,c=`getChannel(getA(${i.join()}), ${a})`;this.userCode=`
    void main() {
      ${o} rc = getOutputCoords();
      vec4 result = vec4(0.);
      result[0] = ${c};
      if(${l}) {
        result[1] = ${c};
      }
      --${r[this.rank-1]};
      if(++${r[this.rank-2]} < ${s[this.rank-2]}) {
        result[2] = ${c};
        if(${l}) {
          result[3] = ${c};
        }
      }
      setOutput(result);
    }
    `}}function jc(n,t,e){const s=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new UB(n.shape,t):new VB(n.shape,t);return e.runWebGLProgram(s,[n],n.dtype)}function GB(n,t,e,s){const o=t,r=n.shape.length,i=$t(o,n.shape);let a=i;const l=Xt(a,r),c=l!=null;let u=n;c&&(u=jc(n,l,s),a=ne(a.length,r)),Ie("sum",a,r);const[h,d]=me(u.shape,a);let p=h;e&&(p=ee(h,i));const f=U(d),g=U(n.shape)/f,x=et({inputs:{x:u},attrs:{shape:[g,f]},backend:s}),b=Qh(n.dtype),w=Fo(x,b,"sum",s),y=et({inputs:{x:w},attrs:{shape:p},backend:s});return s.disposeIntermediateTensorInfo(x),s.disposeIntermediateTensorInfo(w),c&&s.disposeIntermediateTensorInfo(u),y}function Kc(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s;return GB(o,r,i,e)}const HB={kernelName:cc,backendName:"webgl",kernelFunc:Kc};function _e(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{perm:r}=s,i=e,a=o.shape.length,l=new Array(a);for(let u=0;u<l.length;u++)l[u]=o.shape[r[u]];let c;if(i.shouldExecuteOnCPU([o])){const h=i.texData.get(o.dataId).values,d=tf(h,o.shape,o.dtype,r,l);c=i.makeTensorInfo(l,o.dtype);const p=i.texData.get(c.dataId);p.values=d}else c=jc(o,r,i);return c}const qB={kernelName:zo,backendName:"webgl",kernelFunc:_e};const Pw=1e3;function hl({a:n,b:t,transposeA:e,transposeB:s,backend:o,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:a=0,activation:l=null}){const c=n.shape.length,u=t.shape.length,h=e?n.shape[c-2]:n.shape[c-1],d=s?t.shape[u-1]:t.shape[u-2],p=e?n.shape[c-1]:n.shape[c-2],f=s?t.shape[u-2]:t.shape[u-1],m=n.shape.slice(0,-2),g=t.shape.slice(0,-2),x=U(m),b=U(g),y=gt(n.shape.slice(0,-2),t.shape.slice(0,-2)).concat([p,f]);I(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${n.shape} and ${t.shape} and transposeA=${e} and transposeB=${s} must match.`);const $=e?[x,h,p]:[x,p,h],v=s?[b,f,d]:[b,d,f],N=et({inputs:{x:n},backend:o,attrs:{shape:$}}),T=et({inputs:{x:t},backend:o,attrs:{shape:v}}),k=[N,T],S=Math.max(x,b),C=e?N.shape[1]:N.shape[2],D=r!=null,O=i!=null,P=l==="leakyrelu",M=l!=null?Hr(l,!0):null,V=D||O||P||M!=null;let H;if((p===1||f===1)&&C>Pw&&V===!1){let q=N,j=T;e&&(q=_e({inputs:{x:N},backend:o,attrs:{perm:[0,2,1]}}),k.push(q)),s&&(j=_e({inputs:{x:T},backend:o,attrs:{perm:[0,2,1]}}),k.push(j));const K=f!==1,Y=f===1;let Z=q;K&&(Z=et({inputs:{x:q},backend:o,attrs:{shape:[S,C,1]}}),k.push(Z));const tt=f===1?2:1;let Q=j;Y&&(Q=et({inputs:{x:j},backend:o,attrs:{shape:[S,1,C]}}),k.push(Q));const ot=nf({inputs:{a:Z,b:Q},backend:o});H=Kc({inputs:{x:ot},backend:o,attrs:{axis:tt,keepDims:!0}}),k.push(ot)}else{const q=Ke(n.dtype,t.dtype),j=new Mw($,v,[S,p,f],e,s,D,M,O,P),K=[N,T];if(r!=null&&K.push(r),O&&K.push(i),P){const Y=o.makeTensorInfo([],"float32",as(a,"float32"));K.push(Y),k.push(Y)}H=o.runWebGLProgram(j,K,q)}const W=et({inputs:{x:H},backend:o,attrs:{shape:y}});k.push(H);for(const q of k)o.disposeIntermediateTensorInfo(q);return W}function jB(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r,bias:i,preluActivationWeights:a}=t,{transposeA:l,transposeB:c,activation:u,leakyreluAlpha:h}=s;return hl({a:o,b:r,transposeA:l,transposeB:c,backend:e,bias:i,preluActivationWeights:a,leakyreluAlpha:h,activation:u})}const KB={kernelName:Wa,backendName:"webgl",kernelFunc:jB};const Rm="return abs(x);";function XB(n){const{inputs:t,backend:e}=n,{x:s}=t;if(e.shouldExecuteOnCPU([s])&&s.dtype!=="complex64"){const r=e.texData.get(s.dataId),i=Ew(r.values);return e.makeTensorInfo(s.shape,s.dtype,i)}let o;return B().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new Is(s.shape,Rm):o=new zn(s.shape,Rm),e.runWebGLProgram(o,[s],s.dtype)}const YB={kernelName:gl,backendName:"webgl",kernelFunc:XB};const ZB=yn+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return acos(x);
`,JB=St({opSnippet:ZB}),QB={kernelName:Xr,backendName:"webgl",kernelFunc:JB};const tV=yn+`
  if (x < 1.0) return NAN;
return log(x + sqrt(x * x - 1.0));`,eV=St({opSnippet:tV}),nV={kernelName:Yr,backendName:"webgl",kernelFunc:eV};const Am="return a + b;",sV=ve({opSnippet:Am,packedOpSnippet:Am,supportsComplex:!0,cpuKernelImpl:dz}),oV={kernelName:tr,backendName:"webgl",kernelFunc:sV};class rV{constructor(t,e){this.outputShape=[],this.outputShape=t,this.variableNames=e.map((r,i)=>`T${i}`);const s=[];this.variableNames.forEach(r=>{s.push(`float v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        float result = ${o};
        setOutput(result);
      }
    `}}class iV{constructor(t,e){this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.variableNames=e.map((r,i)=>`T${i}`);const s=[];this.variableNames.forEach(r=>{s.push(`vec4 v${r} = get${r}AtOutCoords();`)});const o=this.variableNames.map(r=>`v${r}`).join(" + ");this.userCode=`
      void main() {
        ${s.join(`
        `)}

        vec4 result = ${o};
        setOutput(result);
      }
    `}}function Pa(n){const{inputs:t,backend:e}=n,s=t;if(s.length===1)return Ye({inputs:{x:s[0]},backend:e});if(s.length>B().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER")){const l=Math.floor(s.length/2),c=Pa({inputs:s.slice(0,l),backend:e}),u=Pa({inputs:s.slice(l),backend:e});return Pa({inputs:[c,u],backend:e})}const o=s.map(l=>l.dtype).reduce((l,c)=>Ke(l,c)),r=s.map(l=>l.shape),a=B().getBool("WEBGL_PACK")?new iV(s[0].shape,r):new rV(s[0].shape,r);return e.runWebGLProgram(a,s,o)}const aV={kernelName:sh,backendName:"webgl",kernelFunc:Pa};function lV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=$t(r,o.shape);let c=l;const u=Xt(c,a);let h=o;u!=null&&(h=_e({inputs:{x:o},backend:e,attrs:{perm:u}}),c=ne(c.length,a)),Ie("all",c,a);const[d,p]=me(h.shape,c),f=U(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=Fo(m,m.dtype,"all",e);let x;if(i){const b=ee(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const cV={kernelName:oh,backendName:"webgl",kernelFunc:lV};function uV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=$t(r,o.shape);let c=l;const u=Xt(c,a);let h=o;u!=null&&(h=_e({inputs:{x:o},backend:e,attrs:{perm:u}}),c=ne(c.length,a)),Ie("any",c,a);const[d,p]=me(h.shape,c),f=U(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=Fo(m,m.dtype,"any",e);let x;if(i){const b=ee(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const hV={kernelName:rh,backendName:"webgl",kernelFunc:uV};class dV{constructor(t,e,s){this.variableNames=["A"];const{windowSize:o,batchSize:r,outSize:i}=t;s||this.variableNames.push("bestIndicesA"),this.outputShape=[r,i];const a=e==="max"?">":"<",l=s?"inOffset + i;":"round(getBestIndicesA(batch, inOffset + i));";this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = outIdx * ${o};

        int bestIndex = inOffset;
        float bestValue = getA(batch, bestIndex);

        for (int i = 0; i < ${o}; i++) {
          int inIdx = ${l};
          float candidate = getA(batch, inIdx);
          if (candidate ${a} bestValue) {
            bestValue = candidate;
            bestIndex = inIdx;
          }
        }
        setOutput(float(bestIndex));
      }
    `}}class pV{constructor(t,e,s,o){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,I(t.length>2,()=>`Packed arg${s.charAt(0).toUpperCase()+s.slice(1)} supports only inputs with rank above 2.`);const r=t[t.length-1],i=Math.ceil(r/e);this.outputShape=t.slice(0,-1),i>1&&this.outputShape.push(i),o||this.variableNames.push("bestIndicesA");const a=this.outputShape,l=a.length,c=Ot(l),u=Fe("coords",l);let h,d;if(i===1){d=l+1;const T=Ot(d);h=`
        ${T} sourceLocR = ${T}(${u.join()}, 0);
        ++${u[l-1]};
        ${T} sourceLocG = ${T}(${u.join()}, 0);
        ++${u[l-2]};
        ${T} sourceLocA = ${T}(${u.join()}, 0);
        --${u[l-1]};
        ${T} sourceLocB = ${T}(${u.join()}, 0);
        --${u[l-2]};`}else d=l,h=`
        ${c} sourceLocR = coords;
        ++${u[l-1]};
        ${c} sourceLocG = coords;
        ++${u[l-2]};
        ${c} sourceLocA = coords;
        --${u[l-1]};
        ${c} sourceLocB = coords;
        --${u[l-2]};`;const p=["x","y","z","w","u","v"].slice(0,d),f="."+p[d-1],m=p.map(T=>"int "+T),g=Fe("sourceLocR",d-1).concat("inIdx.r"),x=Fe("sourceLocG",d-1).concat("inIdx.g"),b=Fe("sourceLocB",d-1).concat("inIdx.b"),w=Fe("sourceLocA",d-1).concat("inIdx.a"),y=s==="max"?"greaterThan":"lessThan",$=o?"":`
          inIdx = round(vec4(getBestIndicesAChannel(${g.join()}),
                             getBestIndicesAChannel(${x.join()}),
                             getBestIndicesAChannel(${b.join()}),
                             getBestIndicesAChannel(${w.join()})));`,v=`vec4(
            getAChannel(${g.join()}),
            hasNextCol ? getAChannel(${x.join()}) : 0.,
            hasNextRow ? getAChannel(${b.join()}) : 0.,
            hasNextRow && hasNextCol ? getAChannel(${w.join()}) : 0.)`,N=o?"":`
      float getBestIndicesAChannel(${m.join()}) {
        return getChannel(getBestIndicesA(${p.join()}),
                                          vec2(${p.slice(-2).join()}));
      }`;this.userCode=`
      float getAChannel(${m.join()}) {
        return getChannel(getA(${p.join()}),
                               vec2(${p.slice(-2).join()}));
      }
      ${N}
      void main() {
        ${c} coords = getOutputCoords();
        bool hasNextCol = ${u[l-1]} < ${a[l-1]-1};
        bool hasNextRow = ${u[l-2]} < ${a[l-2]-1};
        ${h}
        ivec4 srcIdx = ivec4(sourceLocR${f}, sourceLocG${f},
          sourceLocB${f}, sourceLocA${f}) * ${e};
        ivec4 inIdx = srcIdx;
        vec4 bestIndex = vec4(inIdx);
        vec4 bestValue = ${v};

        for (int i = 0; i < ${e}; i++) {
          inIdx = srcIdx;
          ${$}
          vec4 candidate = ${v};
          bvec4 nan = isnan(candidate);
          bvec4 replace = bvec4(
            vec4(${y}(candidate, bestValue)) * (vec4(1.0) - vec4(nan)));

          bestValue = vec4(replace.x  ? candidate.x : bestValue.x,
                           replace.y  ? candidate.y : bestValue.y,
                           replace.z  ? candidate.z : bestValue.z,
                           replace.w  ? candidate.w : bestValue.w);
          bestIndex = mix(bestIndex, vec4(inIdx), vec4(replace));
          srcIdx++;
        }
        setOutput(bestIndex);
      }
    `}}function zw(n,t,e,s=null){let o=t.shape[0],r=t.shape[1];s!=null&&(o=s.shape[0],r=s.shape[1]);const i=Tc(r),a={windowSize:i,inSize:r,batchSize:o,outSize:Math.ceil(r/i)},l=new dV(a,e,s==null),c=[t];s!=null&&c.push(s);const u=n.runWebGLProgram(l,c,"int32");if(u.shape[1]===1)return u;const h=zw(n,t,e,u);return n.disposeIntermediateTensorInfo(u),h}function Bw(n,t,e,s=null){const o=s!=null?s.shape:t.shape,r=o[o.length-1],i=Tc(r),a=new pV(o,i,e,s==null),l=s==null?[t]:[t,s],c=n.runWebGLProgram(a,l,"int32");if(c.shape.length===t.shape.length){const u=Bw(n,t,e,c);return n.disposeIntermediateTensorInfo(c),u}return c}function Vw(n,t,e,s){const o=[e];if(Ie("arg"+s.charAt(0).toUpperCase()+s.slice(1),o,t.shape.length),!B().getBool("WEBGL_PACK_REDUCE")||t.shape.length<=2){const r=[],i=n.texData.get(t.dataId),a=i!==null&&i.isPacked;let l=t;a&&(l=n.unpackTensor(t),r.push(l));const[c,u]=me(l.shape,o),h=U(u),d=et({inputs:{x:l},backend:n,attrs:{shape:[-1,h]}});r.push(d);const p=zw(n,d,s);r.push(p);const f=et({inputs:{x:p},backend:n,attrs:{shape:c}});return r.forEach(m=>n.disposeIntermediateTensorInfo(m)),f}return Bw(n,t,s)}function fV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;let i=$t(r,o.shape);const a=Xt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=_e({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=ne(i.length,l.shape.length)),Ie("argMax",[i[0]],l.shape.length);const u=Vw(e,l,i[0],"max");return c.forEach(h=>e.disposeIntermediateTensorInfo(h)),u}const mV={kernelName:xl,backendName:"webgl",kernelFunc:fV};function gV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r}=s;let i=$t(r,o.shape);const a=Xt(i,o.shape.length);let l=o;const c=[];a!=null&&(l=_e({inputs:{x:o},backend:e,attrs:{perm:a}}),c.push(l),i=ne(i.length,l.shape.length)),Ie("argMin",[i[0]],l.shape.length);const u=Vw(e,l,i[0],"min");return c.forEach(h=>e.disposeIntermediateTensorInfo(h)),u}const xV={kernelName:bl,backendName:"webgl",kernelFunc:gV};const bV=yn+`
  if (abs(x) > 1.) {
    return NAN;
  }
  return asin(x);
`,yV=St({opSnippet:bV}),wV={kernelName:Zr,backendName:"webgl",kernelFunc:yV};const $V=yn+"return log(x + sqrt(x * x + 1.0));",CV=St({opSnippet:$V}),IV={kernelName:Jr,backendName:"webgl",kernelFunc:CV};const vV=yn+`
  return atan(x);
`,kV=St({opSnippet:vV}),SV={kernelName:Qr,backendName:"webgl",kernelFunc:kV};const NV=ef+`
  return atan(a, b);
`,TV=`
  vec4 result = atan(a, b);
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Do+`
  return result;
`,EV=ve({opSnippet:NV,packedOpSnippet:TV}),RV={kernelName:ei,backendName:"webgl",kernelFunc:EV};const AV=yn+`
  if ((x < -1.0) || (x > 1.0)) return NAN;
return (log(1.0 + x) - log(1.0 - x)) / 2.0;`,DV=St({opSnippet:AV}),FV={kernelName:ti,backendName:"webgl",kernelFunc:DV};class qr{constructor(t,e,s,o=!1,r=!1){if(this.variableNames=["x"],e==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=t.filterWidth,a=t.strideHeight,l=t.strideWidth,c=t.dilationHeight,u=t.dilationWidth,h=t.effectiveFilterHeight,d=t.effectiveFilterWidth,p=t.padInfo.top,f=t.padInfo.left;this.outputShape=t.outShape;const m=e==="avg",g=`((batch  * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + d`,x=`(xR * ${t.inWidth} + xC) * ${t.inChannels} + d`;let b="0.0";if(m||(b="-1.0 / 1e-20"),s){this.userCode=`
        const ivec2 strides = ivec2(${a}, ${l});
        const ivec2 pads = ivec2(${p}, ${f});

        void main() {
          ivec4 coords = getOutputCoords();
          int batch = coords[0];
          int d = coords[3];

          ivec2 xRCCorner = coords.yz * strides - pads;
          int xRCorner = xRCCorner.x;
          int xCCorner = xRCCorner.y;

          // max/min x(?, ?, d) to get y(yR, yC, d).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;
          float avgValue = 0.0;

          for (int wR = 0; wR < ${h};
              wR += ${c}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${d};
                wC += ${u}) {
              int xC = xCCorner + wC;

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              float value = getX(batch, xR, xC, d);

              // If a min / max value has already been found, use it. If not,
              // use the current value.
              float currMinMaxValue = mix(
                  value, minMaxValue, minMaxValueFound);
              if (value >= currMinMaxValue) {
                minMaxValue = value;
                minMaxValueFound = 1.0;
                minMaxPosition = ${o?r?g:x:`wR * ${d} + wC`};
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const w="max";let y=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="avg"&&(y="avgValue / max(count, 1.0)");const $=Math.floor(i/4)*4,v=i%4,N=`
      if (${m}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${w}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec2 strides = ivec2(${a}, ${l});
      const ivec2 pads = ivec2(${p}, ${f});
      const float initializationValue = ${b};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xR, int xC, int d) {
        if (xC < 0 || xC >= ${t.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xR, xC, d);
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d = coords[3];

        ivec2 xRCCorner = coords.yz * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // max/min x(?, ?, d) to get y(yR, yC, d).
        // ? = to be determined
        vec4 minMaxValue = vec4(${b});
        float avgValue = 0.0;
        count = 0.0;

        for (int wR = 0; wR < ${h};
            wR += ${c}) {
          int xR = xRCorner + wR;

          if (xR < 0 || xR >= ${t.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${$}; wC += 4) {
            int xC = xCCorner + wC * ${u};

            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              getValue(batch, xR, xC + 3 * ${u}, d)
            );

            ${N}
          }

          int xC = xCCorner + ${$};
          if (${v===1}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              initializationValue,
              initializationValue,
              initializationValue
            );

            ${N}
          } else if (${v===2}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              initializationValue,
              initializationValue
            );

            ${N}
          } else if (${v===3}) {
            vec4 values = vec4(
              getValue(batch, xR, xC, d),
              getValue(batch, xR, xC + ${u}, d),
              getValue(batch, xR, xC + 2 * ${u}, d),
              initializationValue
            );

            ${N}
          }
        }
        setOutput(${y});
      }
    `}}class sf{constructor(t,e,s,o=!1,r=!1){if(this.variableNames=["x"],e==="avg"&&s)throw new Error("Cannot compute positions for average pool.");const i=t.filterWidth,a=t.strideDepth,l=t.strideHeight,c=t.strideWidth,u=t.dilationDepth,h=t.dilationHeight,d=t.dilationWidth,p=t.effectiveFilterDepth,f=t.effectiveFilterHeight,m=t.effectiveFilterWidth,g=t.padInfo.front,x=t.padInfo.top,b=t.padInfo.left;this.outputShape=t.outShape;const w=e==="avg";let y="0.0";if(w||(y="-1.0 / 1e-20"),s){this.userCode=`
        const ivec3 strides =
            ivec3(${a}, ${l}, ${c});
        const ivec3 pads = ivec3(${g}, ${x}, ${b});

        void main() {
          ivec5 coords = getOutputCoords();
          int batch = coords.x;
          int ch = coords.u;

          ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
          int xDCorner = xCorner.x;
          int xRCorner = xCorner.y;
          int xCCorner = xCorner.z;

          // max/min x(?, ?, ?, ch) to get y(yD, yR, yC, ch).
          // ? = to be determined
          float minMaxValue = 0.0;
          float minMaxValueFound = 0.0;
          int minMaxPosition = 0;

          for (int wD = 0; wD < ${p};
              wD += ${u}) {
            int xD = xDCorner + wD;

            if (xD < 0 || xD >= ${t.inDepth}) {
              continue;
            }

            for (int wR = 0; wR < ${f};
                wR += ${h}) {
              int xR = xRCorner + wR;

              if (xR < 0 || xR >= ${t.inHeight}) {
                continue;
              }

              for (int wC = 0; wC < ${m};
                  wC += ${d}) {
                int xC = xCCorner + wC;

                if (xC < 0 || xC >= ${t.inWidth}) {
                  continue;
                }

                float value = getX(batch, xD, xR, xC, ch);

                // If a min / max value has already been found, use it. If not,
                // use the current value.
                float currMinMaxValue = mix(
                    value, minMaxValue, minMaxValueFound);
                if (value >= currMinMaxValue) {
                  minMaxValue = value;
                  minMaxValueFound = 1.0;
                  minMaxPosition = ${o?r?`(((batch * ${t.inDepth} + xD) * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`((xD * ${t.inHeight} + xR) * ${t.inWidth} + xC) * ${t.inChannels} + ch`:`wD * ${f} * ${m} +
                      wR * ${m} + wC`};
                }
              }
            }
          }
          setOutput(float(minMaxPosition));
        }
      `;return}const $="max";let v=`${e}(${e}(${e}(minMaxValue[0], minMaxValue[1]), minMaxValue[2]), minMaxValue[3])`;e==="avg"&&(v="avgValue / max(count, 1.0)");const N=Math.floor(i/4)*4,T=i%4,k=`
      if (${w}) {
        avgValue += dot(values, ones);
      } else {
        minMaxValue = ${$}(values, minMaxValue);
      }
    `;this.userCode=`
      const ivec3 strides =
        ivec3(${a}, ${l}, ${c});
      const ivec3 pads = ivec3(${g}, ${x}, ${b});
      const float initializationValue = ${y};
      const vec4 ones = vec4(1.0, 1.0, 1.0, 1.0);

      float count = 0.0;

      float getValue(int batch, int xD, int xR, int xC, int ch) {
        if (xC < 0 || xC >= ${t.inWidth}) {
          return initializationValue;
        }
        count += 1.0;
        return getX(batch, xD, xR, xC, ch);
      }

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 xCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xDCorner = xCorner.x;
        int xRCorner = xCorner.y;
        int xCCorner = xCorner.z;

        // max/min x(?, ?, ?, d) to get y(yD, yR, yC, ch).
        // ? = to be determined
        vec4 minMaxValue = vec4(${y});
        float avgValue = 0.0;
        count = 0.0;

        for (int wD = 0; wD < ${p};
            wD += ${u}) {
          int xD = xDCorner + wD;

          if (xD < 0 || xD >= ${t.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${f};
            wR += ${h}) {
            int xR = xRCorner + wR;

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${N}; wC += 4) {
              int xC = xCCorner + wC * ${d};

              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                getValue(batch, xD, xR, xC + 3 * ${d}, ch)
              );

              ${k}
            }

            int xC = xCCorner + ${N};
            if (${T===1}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                initializationValue,
                initializationValue,
                initializationValue
              );

              ${k}
            } else if (${T===2}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                initializationValue,
                initializationValue
              );

              ${k}
            } else if (${T===3}) {
              vec4 values = vec4(
                getValue(batch, xD, xR, xC, ch),
                getValue(batch, xD, xR, xC + ${d}, ch),
                getValue(batch, xD, xR, xC + 2 * ${d}, ch),
                initializationValue
              );

              ${k}
            }
          }
        }
        setOutput(${v});
      }
    `}}function _V(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;ar(o,"avgPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;I(Te(i,c),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=xn(o.shape,r,i,c,a,l);if(u.filterWidth===1&&u.filterHeight===1&&Tt(u.inShape,u.outShape))return Ye({inputs:{x:o},backend:e});const h=new qr(u,"avg",!1);return e.runWebGLProgram(h,[o],"float32")}const OV={kernelName:yl,backendName:"webgl",kernelFunc:_V};function LV(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dimRoundingMode:l,dataFormat:c}=s,u=[1,1,1],h=cs(o.shape,r,i,u,a,l,c),d=new sf(h,"avg",!1);return e.runWebGLProgram(d,[o],"float32")}const MV={kernelName:wl,backendName:"webgl",kernelFunc:LV};class PV{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=t.dilationHeight,a=t.dilationWidth,l=t.effectiveFilterHeight,c=t.effectiveFilterWidth,u=l-1-t.padInfo.top,h=c-1-t.padInfo.left,d=1/(e*s);this.userCode=`
      const ivec2 pads = ivec2(${u}, ${h});
      const float avgMultiplier = float(${d});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${l};
            wR += ${i}) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${c};
            wC+= ${a}) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);

            dotProd += dyValue * avgMultiplier;
          }
        }
        setOutput(dotProd);
      }
    `}}class zV{constructor(t){this.variableNames=["dy"],this.outputShape=t.inShape;const e=t.filterDepth,s=t.filterHeight,o=t.filterWidth,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=t.dilationDepth,c=t.dilationHeight,u=t.dilationWidth,h=t.effectiveFilterDepth,d=t.effectiveFilterHeight,p=t.effectiveFilterWidth,f=h-1-t.padInfo.front,m=d-1-t.padInfo.top,g=p-1-t.padInfo.left,x=1/(e*s*o);this.userCode=`
      const ivec3 pads = ivec3(${f}, ${m}, ${g});
      const float avgMultiplier = float(${x});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, d) with pos mask(:, :, :, ch) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${h};
            wD += ${l}) {
          float dyD = float(dyDCorner + wD) / ${r}.0;

          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${d};
              wR += ${c}) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${p};
                wC += ${u}) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);

              dotProd += dyValue * avgMultiplier;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function BV(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r,{filterSize:a,strides:l,pad:c,dimRoundingMode:u}=s,h=[1,1,1],d=cs(i.shape,a,l,h,c,u),p=new zV(d);return e.runWebGLProgram(p,[o],i.dtype)}const VV={kernelName:ah,backendName:"webgl",kernelFunc:BV};function WV(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r;ar([o,r],"avgPoolGrad");const{filterSize:a,strides:l,pad:c}=s,u=xn(i.shape,a,l,1,c),h=new PV(u);return e.runWebGLProgram(h,[o],i.dtype)}const UV={kernelName:ih,backendName:"webgl",kernelFunc:WV};function GV(n){const{inputs:t,backend:e,attrs:s}=n,{a:o,b:r}=t,{transposeA:i,transposeB:a}=s;return hl({a:o,b:r,transposeA:i,transposeB:a,backend:e})}const HV={kernelName:$l,backendName:"webgl",kernelFunc:GV};class qV{constructor(t,e,s,o,r,i){this.outputShape=[],this.variableNames=["x","mean","variance"],gt(t,e),gt(t,s);let a="0.0";o!=null&&(gt(t,o),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let l="1.0";r!=null&&(gt(t,r),this.variableNames.push("scale"),l="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`
      void main() {
        float x = getXAtOutCoords();
        float mean = getMeanAtOutCoords();
        float variance = getVarianceAtOutCoords();
        float offset = ${a};
        float scale = ${l};
        float inv = scale * inversesqrt(variance + float(${i}));
        setOutput(dot(vec3(x, -mean, offset), vec3(inv, inv, 1)));
      }
    `}}class jV{constructor(t,e,s,o,r,i){this.packedInputs=!0,this.packedOutput=!0,this.variableNames=["x","mean","variance"],gt(t,e),gt(t,s);let a="vec4(0.0)";o!=null&&(gt(t,o),this.variableNames.push("offset"),a="getOffsetAtOutCoords()");let l="vec4(1.0)";r!=null&&(gt(t,r),this.variableNames.push("scale"),l="getScaleAtOutCoords()"),this.outputShape=t,this.userCode=`
      void main() {
        vec4 offset = ${a};
        vec4 scale = ${l};

        vec4 x = getXAtOutCoords();
        vec4 mean = getMeanAtOutCoords();
        vec4 variance = getVarianceAtOutCoords();

        vec4 inv = scale * inversesqrt(variance + vec4(${i}));

        setOutput((x - mean) * inv + offset);
      }
    `}}const KV=({inputs:n,backend:t,attrs:e})=>{const{x:s,mean:o,variance:r,offset:i,scale:a}=n;I(o.shape.length===r.shape.length,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),I(i==null||o.shape.length===i.shape.length,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),I(a==null||o.shape.length===a.shape.length,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");let{varianceEpsilon:l}=e;l==null&&(l=.001);const c=[s,o,r];let u=null;i!=null&&(u=i.shape,c.push(i));let h=null;a!=null&&(h=a.shape,c.push(a));const d=B().getBool("WEBGL_PACK_NORMALIZATION")?new jV(s.shape,o.shape,r.shape,u,h,l):new qV(s.shape,o.shape,r.shape,u,h,l);return t.runWebGLProgram(d,c,c[0].dtype)},XV={kernelName:Fl,backendName:"webgl",kernelFunc:KV};class YV{constructor(t){this.variableNames=["source"],this.outputShape=t,this.rank=t.length;const e=Ot(this.rank);this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const s=ZV(this.rank);let o;const r=t.map((i,a)=>`sourceLoc.${ju[a]} = start[${a}] + coords.${ju[a]};`);o=`
        ${e} sourceLoc;
        ${e} coords = getOutputCoords();
        ${r.join(`
`)}
      `,this.userCode=`
      void main() {
        ${o}
        setOutput(getSource(${s}));
      }
    `}}const ju=["x","y","z","w","u","v"];function ZV(n){if(n===1)return"sourceLoc";if(n<=6)return ju.slice(0,n).map(t=>"sourceLoc."+t).join(",");throw Error(`Slicing for rank ${n} is not yet supported`)}class JV{constructor(t){this.variableNames=["source"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=t,this.rank=t.length,this.customUniforms=[{name:"start",arrayIndex:this.rank,type:"int"}];const e=Ot(this.rank),s=Fe("coords",this.rank),o=Fe("sourceLoc",this.rank),r=this.rank===1?"sourceLoc":`vec2(${o.slice(-2).join()})`,i=`getChannel(getSource(${o.join()}), ${r})`,a=`
      result.x = ${i};
      if (++${s[this.rank-1]} < ${t[this.rank-1]}) {
        ++${o[this.rank-1]};
        result.y = ${i};
        --${o[this.rank-1]};
      }
    `,l=this.rank===1?"":`
      --${s[this.rank-1]};
      if (++${s[this.rank-2]} < ${t[this.rank-2]}) {
        ++${o[this.rank-2]};
        result.z = ${i};
        if (++${s[this.rank-1]} < ${t[this.rank-1]}) {
          ++${o[this.rank-1]};
          result.w = ${i};
        }
      }
    `,c=this.rank<=4?`sourceLoc = coords +
            ${e}(${t.map((u,h)=>`start[${h}]`).join()});`:t.map((u,h)=>`${o[h]} = ${s[h]} + start[${h}];`).join(`
`);this.userCode=`
      void main() {
        ${e} coords = getOutputCoords();
        ${e} sourceLoc;
        ${c}
        vec4 result = vec4(0.);
        ${a}
        ${l}
        setOutput(result);
      }
    `}}function QV(n,t,e,s){const o=s.texData.get(n.dataId),r=s.makeTensorInfo(e,n.dtype),i=s.texData.get(r.dataId);Object.assign(i,o),i.refCount=1,i.shape=e,i.dtype=n.dtype;let a=Gd(t,ut(n.shape));o.slice&&(a+=o.slice.flatOffset),i.slice={flatOffset:a,origDataId:o.slice&&o.slice.origDataId||n.dataId};const l=s.dataRefCount.get(i.slice.origDataId)||1;return s.dataRefCount.set(i.slice.origDataId,l+1),r}function fr(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,size:i}=s,[a,l]=Nc(o,r,i);if(Vd(o,a,l),U(l)===0)return e.makeTensorInfo(l,o.dtype,[]);if(e.shouldExecuteOnCPU([o])||o.dtype==="string"){const h=e.texData.get(o.dataId),d=Gz(h.values,a,l,o.shape,o.dtype);return e.makeTensorInfo(l,o.dtype,d)}const{isPacked:c}=e.texData.get(o.dataId),u=Ud(o.shape,a,l);if(c||!u){const h=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new JV(l):new YV(l),d=[a];return e.runWebGLProgram(h,[o],o.dtype,d)}return e.uploadToGPU(o.dataId),QV(o,a,l,e)}const tW={kernelName:lc,backendName:"webgl",kernelFunc:fr};const eW=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,crops:i}=s;I(o.shape.length<=4,()=>"batchToSpaceND for rank > 4 with a WebGL backend not implemented yet");const a=r.reduce((b,w)=>b*w),l=Zi(o.shape,r,a),c=Ji(l.length,r.length),u=Qi(o.shape,r,a),h=Xd(i,r.length),d=Yd(u,i,r.length),p=[],f=et({inputs:{x:o},backend:e,attrs:{shape:l}}),m=_e({inputs:{x:f},backend:e,attrs:{perm:c}}),g=et({inputs:{x:m},backend:e,attrs:{shape:u}}),x=fr({inputs:{x:g},backend:e,attrs:{begin:h,size:d}});return p.push(f),p.push(m),p.push(g),p.forEach(b=>e.disposeIntermediateTensorInfo(b)),x},nW={kernelName:Cl,backendName:"webgl",kernelFunc:eW};function sW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i}=s,a=e.readSync(o.dataId),l=e.readSync(r.dataId),c=Tw(a,l,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,c)}const oW={kernelName:lh,backendName:"webgl",kernelFunc:sW};const rW=`
  int r = int(a.r) & int(b.r);
  int g = int(a.g) & int(b.g);
  int rb = int(a.b) & int(b.b);
  int ra = int(a.a) & int(b.a);
  return vec4(r, g, rb, ra);
`,iW=`
  return float(int(a.r) & int(b.r));
`;function aW(n){const{inputs:t,backend:e}=n,{a:s,b:o}=t,r=B().getBool("WEBGL_PACK_BINARY_OPERATIONS"),i=B().getNumber("WEBGL_VERSION");if(e.shouldExecuteOnCPU([s,o])||i===1){const l=e.texData.get(s.dataId).values,c=e.texData.get(o.dataId).values,[u,h]=fz(s.shape,o.shape,l,c,s.dtype),d=e.makeTensorInfo(h,s.dtype),p=e.texData.get(d.dataId);return p.values=u,d}let a;return r?a=new dr(rW,s.shape,o.shape,!1):a=new Io(iW,s.shape,o.shape),e.runWebGLProgram(a,[s,o],s.dtype)}const lW={kernelName:ch,backendName:"webgl",kernelFunc:aW};function cW(n){const{inputs:t,backend:e}=n,{s0:s,s1:o}=t,r=e.readSync(s.dataId),i=e.readSync(o.dataId),a=gt(Array.from(r),Array.from(i));return e.makeTensorInfo([a.length],"int32",Int32Array.from(a))}const uW={kernelName:tg,backendName:"webgl",kernelFunc:cW};const hW="return float(a != b);",Ww=ve({opSnippet:hW,cpuKernelImpl:Oz,dtype:"bool"}),dW={kernelName:Yl,backendName:"webgl",kernelFunc:Ww};function ha(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.texData.get(s.dataId);return Ye({inputs:{x:o.complexTensorInfos.real},backend:e})}const pW={kernelName:Oh,backendName:"webgl",kernelFunc:ha};const fW="return float(int(x));";function mW(n,t){const e=new zn(n.shape,fW),s=t.runWebGLProgram(e,[n],"int32");return{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}function Ku(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dtype:r}=s;if(r==="complex64"){if(o.dtype==="complex64")return Ye({inputs:{x:o},backend:e});const i=pe(o.shape),a=Ku({inputs:{x:o},backend:e,attrs:{dtype:"float32"}}),l=Gs({inputs:{real:a,imag:i},backend:e});return i.dispose(),e.disposeIntermediateTensorInfo(a),l}if(o.dtype==="complex64"){const i=ha({inputs:{input:o},backend:e}),a=Ku({inputs:{x:i},backend:e,attrs:{dtype:r}});return e.disposeIntermediateTensorInfo(i),a}if(!th(o.dtype,r)){const i=Ye({inputs:{x:o},backend:e});return{dataId:i.dataId,shape:i.shape,dtype:r}}if(e.shouldExecuteOnCPU([o])){const i=e.texData.get(o.dataId).values,[a,l,c]=mz(i,o.shape,o.dtype,r);return e.makeTensorInfo(a,l,c)}if(r==="int32")return mW(o,e);if(r==="bool"){const i=e.makeTensorInfo([],"bool",we("bool",1)),l=Ww({inputs:{a:o,b:i},backend:e});return e.disposeIntermediateTensorInfo(i),l}throw new Error(`Error in Cast: failed to cast ${o.dtype} to ${r}`)}const gW={kernelName:ni,backendName:"webgl",kernelFunc:Ku};const Dm="return ceil(x);",xW=St({opSnippet:Dm,packedOpSnippet:Dm,cpuKernelImpl:gz}),bW={kernelName:si,backendName:"webgl",kernelFunc:xW};class yW{constructor(t){this.variableNames=["A"],this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=t,this.userCode=`

      void main() {
        float value = getAAtOutCoords();
        if (isnan(value)) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, minVal, maxVal));
      }
    `}}class wW{constructor(t){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"minVal",type:"float"},{name:"maxVal",type:"float"}],this.outputShape=t,this.userCode=`
      void main() {
        vec4 value = getAAtOutCoords();

        if (any(isnan(value))) {
          setOutput(value);
          return;
        }

        setOutput(clamp(value, vec4(minVal), vec4(maxVal)));
      }
    `}}function $W(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{clipValueMin:r,clipValueMax:i}=s;let a;B().getBool("WEBGL_PACK_CLIP")?a=new wW(o.shape):a=new yW(o.shape);const l=[[r],[i]];return e.runWebGLProgram(a,[o],o.dtype,l)}const CW={kernelName:oi,backendName:"webgl",kernelFunc:$W};class IW{constructor(t){this.variableNames=["real","imag"],this.outputShape=t,this.userCode=`
      void main() {
        float re = abs(getRealAtOutCoords());
        float im = abs(getImagAtOutCoords());
        float mx = max(re, im);

        // sadly the length function in glsl is not underflow-safe
        // (at least not on Intel GPUs). So the safe solution is
        // to ensure underflow-safety in all cases.
        setOutput(
          mx == 0.0 ? 0.0 : mx * length(vec2(1, min(re, im)/mx))
        );
      }
    `}}function Fm(n,t){return{dataId:t.dataId,dtype:t.dtype,shape:n.shape}}function vW(n){const{inputs:t,backend:e}=n,{x:s}=t,o=e.texData.get(s.dataId),r=new IW(s.shape),i=[Fm(s,o.complexTensorInfos.real),Fm(s,o.complexTensorInfos.imag)];return e.runWebGLProgram(r,i,i[0].dtype)}const kW={kernelName:Il,backendName:"webgl",kernelFunc:vW};class SW{constructor(t){this.outputShape=[],this.outputShape=Vn(t,1),this.variableNames=t.map((i,a)=>`T${a}`);const e=new Array(t.length-1);e[0]=t[0][1];for(let i=1;i<e.length;i++)e[i]=e[i-1]+t[i][1];const s=[`if (yC < ${e[0]}) setOutput(getT0(yR, yC));`];for(let i=1;i<e.length;i++){const a=e[i-1];s.push(`else if (yC < ${e[i]}) setOutput(getT${i}(yR, yC-${a}));`)}const o=e.length,r=e[e.length-1];s.push(`else setOutput(getT${o}(yR, yC-${r}));`),this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int yR = coords.x;
        int yC = coords.y;

        ${s.join(`
        `)}
      }
    `}}class NW{constructor(t,e){this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[],this.outputShape=Vn(t,e);const s=this.outputShape,o=s.length,r=Ot(o),i=Fe("coords",o),a=["x","y","z","w","u","v"].slice(0,o);this.variableNames=t.map((m,g)=>`T${g}`);const l=new Array(t.length-1);l[0]=t[0][e];for(let m=1;m<l.length;m++)l[m]=l[m-1]+t[m][e];const c=a[e],u=a.slice(-2),h=a.join();let d=`if (${c} < ${l[0]}) {
        return getChannel(
            getT0(${h}), vec2(${u.join()}));
        }`;for(let m=1;m<l.length;m++){const g=l[m-1];d+=`
        if (${c} < ${l[m]}  && ${c} >= ${l[m-1]}) {
          return getChannel(
            getT${m}(${Ia(a,c,g)}),
            vec2(${Ia(u,c,g)}));
        }`}const p=l.length,f=l[l.length-1];d+=`
        return getChannel(
          getT${p}(${Ia(a,c,f)}),
          vec2(${Ia(u,c,f)}));`,this.userCode=`
      float getValue(${a.map(m=>"int "+m)}) {
        ${d}
      }

      void main() {
        ${r} coords = getOutputCoords();
        vec4 result = vec4(getValue(${i}), 0., 0., 0.);

        ${i[o-1]} = ${i[o-1]} + 1;
        if (${i[o-1]} < ${s[o-1]}) {
          result.g = getValue(${i});
        }

        ${i[o-2]} = ${i[o-2]} + 1;
        if (${i[o-2]} < ${s[o-2]}) {
          result.a = getValue(${i});
        }

        ${i[o-1]} = ${i[o-1]} - 1;
        if (${i[o-2]} < ${s[o-2]} &&
            ${i[o-1]} < ${s[o-1]}) {
          result.b = getValue(${i});
        }
        setOutput(result);
      }
    `}}function Ia(n,t,e){const s=n.indexOf(t);return n.map((r,i)=>i===s?`${r} - ${e}`:r).join()}function Xc(n){const{inputs:t,backend:e}=n,{input:s}=t,o=e.texData.get(s.dataId);return Ye({inputs:{x:o.complexTensorInfos.imag},backend:e})}const TW={kernelName:Nh,backendName:"webgl",kernelFunc:Xc};function Tr(n,t,e){const s=n[0].dtype;if(s==="complex64"){const p=n.map(b=>ha({inputs:{input:b},backend:e})),f=n.map(b=>Xc({inputs:{input:b},backend:e})),m=Tr(p,t,e),g=Tr(f,t,e),x=Gs({inputs:{real:m,imag:g},backend:e});return p.forEach(b=>e.disposeIntermediateTensorInfo(b)),f.forEach(b=>e.disposeIntermediateTensorInfo(b)),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),x}let o=e.shouldExecuteOnCPU(n);if(s==="string"&&(o=!0),o){const p=n.map(y=>{const v=[-1,U(y.shape.slice(t))];return et({inputs:{x:y},backend:e,attrs:{shape:v}})}),f=p.map(y=>({vals:e.readSync(y.dataId),shape:y.shape})),m=Vn(p.map(y=>y.shape),1),g=p[0].shape[0]===1,x=xz(f,m,s,g),b=Vn(n.map(y=>y.shape),t),w=e.makeTensorInfo(b,s,x);return p.forEach(y=>e.disposeIntermediateTensorInfo(y)),w}const r=n.filter(p=>U(p.shape)>0),i=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")&&r[0].shape.length>1;if(r.length===1){const p=i?new zn(n[0].shape,ys):new Is(n[0].shape,ys);return e.runWebGLProgram(p,n,s)}const a=B().getNumber("WEBGL_MAX_TEXTURES_IN_SHADER");if(r.length>a){const p=[];for(let m=0;m<r.length;m+=a){const g=r.slice(m,m+a);p.push(Tr(g,t,e))}const f=Tr(p,t,e);for(const m of p)e.disposeIntermediateTensorInfo(m);return f}if(i){const p=new NW(r.map(f=>f.shape),t);return e.runWebGLProgram(p,r,s)}const{tensors2D:l,outShape:c}=EW(r,t,e),u=new SW(l.map(p=>p.shape)),h=e.runWebGLProgram(u,l,s);l.forEach(p=>e.disposeIntermediateTensorInfo(p));const d=et({inputs:{x:h},attrs:{shape:c},backend:e});return e.disposeIntermediateTensorInfo(h),d}function EW(n,t,e){const s=Vn(n.map(r=>r.shape),t);return{tensors2D:n.map(r=>et({inputs:{x:r},attrs:{shape:[-1,U(r.shape.slice(t))]},backend:e})),outShape:s}}function Uw(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s,r=$t(o,t[0].shape)[0],i=t.map(c=>c.shape);qd(i,r);const a=Vn(t.map(c=>c.shape),r);if(U(a)===0)return e.makeTensorInfo(a,t[0].dtype,[]);const l=t.filter(c=>U(c.shape)>0);return l.length===1?Ye({inputs:{x:l[0]},backend:e}):Tr(l,r,e)}const RW={kernelName:vl,backendName:"webgl",kernelFunc:Uw};class Gw{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.outputShape=t.outShape;const i=t.padInfo.top,a=t.padInfo.left,l=t.strideHeight,c=t.strideWidth,u=t.dilationHeight,h=t.dilationWidth,d=t.filterHeight,p=t.filterWidth,f=Math.floor(t.inChannels/4)*4,m=t.inChannels%4,g=t.dataFormat==="channelsLast",x=g?1:2,b=g?2:3,w=g?3:1;let y="",$="";s&&(o?y=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?y=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:y=`
          float activation(float x) {
            ${s}
          }
        `,$="result = activation(result);");const v=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${y}

      const ivec2 strides = ivec2(${l}, ${c});
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d2 = coords[${w}];

        ivec2 xRCCorner =
            ivec2(coords[${x}], coords[${b}]) * strides - pads;
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${d}; wR++) {
          int xR = xRCorner + wR * ${u};

          if (xR < 0 || xR >= ${t.inHeight}) {
            continue;
          }

          for (int wC = 0; wC < ${p}; wC++) {
            int xC = xCCorner + wC * ${h};

            if (xC < 0 || xC >= ${t.inWidth}) {
              continue;
            }

            for (int d1 = 0; d1 < ${f}; d1 += 4) {
              vec4 wValues = vec4(
                getW(wR, wC, d1, d2),
                getW(wR, wC, d1 + 1, d2),
                getW(wR, wC, d1 + 2, d2),
                getW(wR, wC, d1 + 3, d2)
              );

              if (${g}) {
                vec4 xValues = vec4(
                  getX(batch, xR, xC, d1),
                  getX(batch, xR, xC, d1 + 1),
                  getX(batch, xR, xC, d1 + 2),
                  getX(batch, xR, xC, d1 + 3)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec4 xValues = vec4(
                  getX(batch, d1, xR, xC),
                  getX(batch, d1 + 1, xR, xC),
                  getX(batch, d1 + 2, xR, xC),
                  getX(batch, d1 + 3, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }
            }

            if (${m===1}) {

              if (${g}) {
                dotProd +=
                    getX(batch, xR, xC, ${f}) *
                    getW(wR, wC, ${f}, d2);
              } else {
                dotProd +=
                    getX(batch, ${f}, xR, xC) *
                    getW(wR, wC, ${f}, d2);
              }

            } else if (${m===2}) {
              vec2 wValues = vec2(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2)
              );

              if (${g}) {
                vec2 xValues = vec2(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec2 xValues = vec2(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            } else if (${m===3}) {
              vec3 wValues = vec3(
                getW(wR, wC, ${f}, d2),
                getW(wR, wC, ${f} + 1, d2),
                getW(wR, wC, ${f} + 2, d2)
              );

              if (${g}) {
                vec3 xValues = vec3(
                  getX(batch, xR, xC, ${f}),
                  getX(batch, xR, xC, ${f} + 1),
                  getX(batch, xR, xC, ${f} + 2)
                );
                dotProd += dot(xValues, wValues);
              } else {
                vec3 xValues = vec3(
                  getX(batch, ${f}, xR, xC),
                  getX(batch, ${f} + 1, xR, xC),
                  getX(batch, ${f} + 2, xR, xC)
                );
                dotProd += dot(xValues, wValues);
              }

            }
          }
        }

        float result = dotProd;
        ${v}
        ${$}
        setOutput(result);
      }
    `}}class AW{constructor(t){this.variableNames=["x","W"],this.outputShape=t.outShape;const e=t.padInfo.front,s=t.padInfo.top,o=t.padInfo.left,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=t.dilationDepth,c=t.dilationHeight,u=t.dilationWidth,h=t.filterDepth,d=t.filterHeight,p=t.filterWidth,f=Math.floor(t.inChannels/4)*4,m=t.inChannels%4;this.userCode=`
      const ivec3 strides = ivec3(${r}, ${i}, ${a});
      const ivec3 pads = ivec3(${e}, ${s}, ${o});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d2 = coords.u;

        ivec3 xFRCCorner = ivec3(coords.y, coords.z, coords.w) * strides - pads;
        int xFCorner = xFRCCorner.x;
        int xRCorner = xFRCCorner.y;
        int xCCorner = xFRCCorner.z;

        // Convolve x(?, ?, ?, d1) with w(:, :, :, d1, d2) to get
        // y(yF, yR, yC, d2). ? = to be determined. : = across all
        // values in that axis.
        float dotProd = 0.0;
        for (int wF = 0; wF < ${h}; wF++) {
          int xF = xFCorner + wF * ${l};

          if (xF < 0 || xF >= ${t.inDepth}) {
            continue;
          }

          for (int wR = 0; wR < ${d}; wR++) {
            int xR = xRCorner + wR * ${c};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int wC = 0; wC < ${p}; wC++) {
              int xC = xCCorner + wC * ${u};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              for (int d1 = 0; d1 < ${f}; d1 += 4) {
                vec4 xValues = vec4(
                  getX(batch, xF, xR, xC, d1),
                  getX(batch, xF, xR, xC, d1 + 1),
                  getX(batch, xF, xR, xC, d1 + 2),
                  getX(batch, xF, xR, xC, d1 + 3)
                );
                vec4 wValues = vec4(
                  getW(wF, wR, wC, d1, d2),
                  getW(wF, wR, wC, d1 + 1, d2),
                  getW(wF, wR, wC, d1 + 2, d2),
                  getW(wF, wR, wC, d1 + 3, d2)
                );

                dotProd += dot(xValues, wValues);
              }

              if (${m===1}) {
                dotProd +=
                  getX(batch, xF, xR, xC, ${f}) *
                  getW(wF, wR, wC, ${f}, d2);
              } else if (${m===2}) {
                vec2 xValues = vec2(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1)
                );
                vec2 wValues = vec2(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2)
                );
                dotProd += dot(xValues, wValues);
              } else if (${m===3}) {
                vec3 xValues = vec3(
                  getX(batch, xF, xR, xC, ${f}),
                  getX(batch, xF, xR, xC, ${f} + 1),
                  getX(batch, xF, xR, xC, ${f} + 2)
                );
                vec3 wValues = vec3(
                  getW(wF, wR, wC, ${f}, d2),
                  getW(wF, wR, wC, ${f} + 1, d2),
                  getW(wF, wR, wC, ${f} + 2, d2)
                );
                dotProd += dot(xValues, wValues);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class Hw{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Re(this.outputShape.length);const i=t.padInfo.left,a=t.strideWidth,l=t.dilationWidth,c=t.filterHeight,u=t.filterWidth,h=u;let d=`
       int xR; int xC; int xCOffset;
       vec4 wTexel; vec4 previous; vec4 final;`;for(let g=0;g<u;g++)d+=`
           vec4 xTexelC${g*2};
           int xTexelC${g*2}Ready;
           vec4 xTexelC${g*2+1};
           int xTexelC${g*2+1}Ready;
           vec4 xC${g};`;d+=`
     for (int r = 0; r < ${c}; r++) {
      for (int d1 = 0; d1 < ${t.inChannels}; d1 += 2) {
       `;for(let g=0;g<u;g++)d+=`
           xTexelC${g*2} = vec4(0.0);
           xTexelC${g*2}Ready = 0;
           xTexelC${g*2+1} = vec4(0.0);
           xTexelC${g*2+1}Ready = 0;
           xC${g} = vec4(0.0);`;d+=`
         xR = xRCorner + r * dilations[0];
         if (xR >=0 && xR < inDims[0]) {
       `;for(let g=0;g<(h+1)/2;g++){const x=g*2;if(d+=`
           xC = xCCorner + ${x*l};
           `,a===1){if(x<u&&(i%2===1?(d+=`
                 xCOffset = xC + 1;
                 if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xCOffset, d1);

                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }
               `,l===1&&x>0?d+=`
                 xC${x} = vec4(xTexelC${x-2}.zw, xTexelC${x}.xy);
                 `:d+=`
                   xCOffset = xC + 1 - 2;

                   if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       previous.zw = vec2(0.0);
                     }

                     xC${x} = vec4(previous.zw, xTexelC${x}.xy);
                   } else {
                     xC${x} = vec4(0.0, 0.0, xTexelC${x}.xy);
                   }
                   `):d+=`
                 if (xC >= 0 && xC < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 xC${x} = xTexelC${x};
                 `,x+1<u)){const b=i%2===0?fl(l):l;l%2===0&&i%2===1||l%2!==0&&i%2!==1?(d+=`
                   xCOffset = xC + imod(pads[1], 2) + ${b};

                   if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                     xTexelC${x+1} = getX(batch, xR, xCOffset, d1);

                     // Need to manually clear unused channels in case
                     // we're reading from recycled texture.
                     if (xCOffset + 1 >= inDims[1]) {
                       xTexelC${x+1}.zw = vec2(0.0);
                     }
                     xTexelC${x+1}Ready = 1;
                   }
                   `,l>1?d+=`
                     xCOffset -= 2;
                     if (xCOffset >= 0 && xCOffset < inDims[1]) {
                      previous = getX(batch, xR, xCOffset, d1);
                      xC${x+1} = vec4(previous.zw, xTexelC${x+1}.xy);
                     } else {
                      xC${x+1} = vec4(0.0, 0.0, xTexelC${x+1}.xy);
                     }
                     `:d+=`
                     xC${x+1} = vec4(xTexelC${x}.zw, xTexelC${x+1}.xy);
                     `):b===1?d+=`
                     xC${x+1} = xTexelC${x};
                     `:d+=`
                     xCOffset = xC + ${b};

                     if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                       xTexelC${x+1} = getX(batch, xR, xCOffset, d1);
                       if (xCOffset + 1 >= inDims[1]) {
                         xTexelC${x+1}.zw = vec2(0.0);
                       }
                       xTexelC${x+1}Ready = 1;
                     }

                     xC${x+1} = xTexelC${x+1};
                     `}}else x<u&&(i%2===1?(d+=`
                 xCOffset = xC + 1 - strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xCOffset, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${x+1}Ready == 0) {
                   xTexelC${x+1} = getX(batch, xR, xC + 1, d1);
                   // Need to manually clear unused channels in case
                   // we're reading from recycled texture.
                   if (xC + 2 >= inDims[1]) {
                     xTexelC${x+1}.zw = vec2(0.0);
                   }
                   xTexelC${x+1}Ready = 1;
                 }

                 xC${x} = vec4(xTexelC${x}.zw, xTexelC${x+1}.zw);
               `,x+1<u&&(d+=`
                   final = vec4(0.0);
                   xCOffset = xC + 1 + strides[1];
                   if(xCOffset >= 0 && xCOffset < inDims[1]) {
                     final = getX(batch, xR, xCOffset, d1);
                   }
                   xC${x+1} = vec4(xTexelC${x+1}.xy, final.xy);
                 `)):(d+=`
                 if(xC >= 0 && xC < inDims[1] && xTexelC${x}Ready == 0) {
                   xTexelC${x} = getX(batch, xR, xC, d1);
                   if (xC + 1 >= inDims[1]) {
                     xTexelC${x}.zw = vec2(0.0);
                   }
                   xTexelC${x}Ready = 1;
                 }

                 xCOffset = xC + strides[1];
                 if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${x+1}Ready == 0) {
                   xTexelC${x+1} = getX(batch, xR, xCOffset, d1);
                   if (xCOffset + 1 >= inDims[1]) {
                     xTexelC${x+1}.zw = vec2(0.);
                   }
                   xTexelC${x+1}Ready = 1;
                 }

                 xC${x} = vec4(
                   xTexelC${x}.xy, xTexelC${x+1}.xy);
               `,x+1<u&&(d+=`
                   xC${x+1} = vec4(xTexelC${x}.zw, xTexelC${x+1}.zw);
                 `)));x<u&&(d+=`
             wTexel = getW(r, ${x}, d1, d2);
             dotProd += xC${x}.xxzz * vec4(wTexel.xy, wTexel.xy);
             if(d1 + 1 < ${t.inChannels}) {
               dotProd += xC${x}.yyww * vec4(wTexel.zw, wTexel.zw);
             }
           `,x+1<u&&(d+=`
               wTexel = getW(r, ${x+1}, d1, d2);
               dotProd += xC${x+1}.xxzz * vec4(wTexel.xy, wTexel.xy);
               if(d1 + 1 < ${t.inChannels}) {
                 dotProd += xC${x+1}.yyww * vec4(wTexel.zw, wTexel.zw);
               }
             `))}d+=`
     }
   `,d+=`
     }
   `,d+=`
     }
   `;let p="",f="";s&&(o?p=`vec4 activation(vec4 a) {
           vec4 b = getPreluActivationWeightsAtOutCoords();
           ${s}
         }`:r?p=`vec4 activation(vec4 a) {
           vec4 b = getLeakyreluAlphaAtOutCoords();
           ${s}
         }`:p=`vec4 activation(vec4 x) {
           ${s}
         }`,f="result = activation(result);");const m=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
       ${p}

       void main() {
         ivec4 coords = getOutputCoords();
         int batch = coords.x;
         ivec2 xRCCorner = coords.yz * strides - pads;
         int d2 = coords.w;
         int xRCorner = xRCCorner.x;
         int xCCorner = xRCCorner.y;

         //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
         vec4 dotProd = vec4(0.000000000000001);

         ${d}

         vec4 result = dotProd - vec4(0.000000000000001);
         ${m}
         ${f}
         setOutput(result);
       }
     `}}class DW{constructor(t,e){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"inputShape",type:"ivec4"},{name:"pad",type:"ivec2"},{name:"stride",type:"ivec2"},{name:"dilation",type:"ivec2"},{name:"inChannels",type:"int"},{name:"itemsPerBlockRow",type:"int"},{name:"outWidth",type:"int"}],this.outputShape=t,this.enableShapeUniforms=Re(this.outputShape.length);const{dataFormat:s}=e,o=Le(),r=s==="channelsLast",i=r?1:2,a=r?2:3,l=this.enableShapeUniforms?"if(blockIndex < outShape[2] && pos < outShape[1]) {":`if(blockIndex < ${t[2]} && pos < ${t[1]}) {`;let c="";for(let u=0;u<=1;u++)for(let h=0;h<=1;h++)c+=`
          blockIndex = rc.z + ${h};
          pos = rc.y + ${u};

          ${l}
            offsetY = int(blockIndex / outWidth) * stride[0] - pad[0];
            d0 = offsetY + dilation[0] * (pos / itemsPerBlockRow);

            if(d0 < inputShape[${i}] && d0 >= 0) {
              // Use custom imod instead mod. On Intel GPU, mod may generate
              // unexpected value.
              // https://github.com/tensorflow/tfjs/issues/5447
              offsetX = imod(blockIndex, outWidth) * stride[1] - pad[1];
              d1 = offsetX + dilation[1] * (imod(pos, itemsPerBlockRow) /
                  inChannels);

              if(d1 < inputShape[${a}] && d1 >= 0) {

                ch = imod(pos, inChannels);

                if (${r}) {
                  innerDims = vec2(d1, ch);
                  result[${u*2+h}] = getChannel(
                    getA(rc.x, d0, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                } else {
                  innerDims = vec2(d0, d1);
                  result[${u*2+h}] = getChannel(
                    getA(rc.x, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                }
              }
            }
          }
        `;this.userCode=`
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0);

        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;

        ${c}

        ${o.output} = result;
      }
    `}}function dl(n,t){const e=n.length;return e>=3?t?[...n.slice(0,-3),n[e-3]*n[e-2],n[e-1]]:[...n.slice(0,-3),n[e-3],n[e-2]*n[e-1]]:!t&&e===1&&n[0]>1?[n[0],1]:null}function qw({x:n,filter:t,convInfo:e,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:i=0,activation:a=null}){const l=n.shape,c=s.texData.get(n.dataId),u=e.inChannels,h=l[0]*l[1]*l[2],d=e.outChannels,p=e.dataFormat==="channelsLast",f=!1,m=!1;let g;const x=[];if(r!=null){const y=dl(r.shape,p);y!=null&&(r=et({inputs:{x:r},backend:s,attrs:{shape:y}}),x.push(r))}if(o!=null){const y=dl(o.shape,p);y!=null&&(o=et({inputs:{x:o},backend:s,attrs:{shape:y}}),x.push(o))}if(!((h===1||d===1)&&u>Pw)&&c.isPacked&&p&&c.texture!=null&&l[2]%2!==0&&Tt(c.shape.slice(-3),l.slice(-3))){const y=l[0]*l[1]*(l[2]+1),$={dataId:n.dataId,shape:[1,y,e.inChannels],dtype:n.dtype},v=c.shape;c.shape=c.shape.slice(),c.shape[c.shape.length-2]++,I(Gr(c.shape,$.shape),()=>`packed reshape ${c.shape} to ${$.shape} isn't free`);const N=et({inputs:{x:t},backend:s,attrs:{shape:[1,e.inChannels,e.outChannels]}});x.push(N);const T=hl({a:$,b:N,backend:s,transposeA:f,transposeB:m,bias:o,activation:a,preluActivationWeights:r,leakyreluAlpha:i}),k=s.texData.get(T.dataId);I(k.isPacked,()=>"batchMatMul result is expected to be packed"),c.shape=v,k.shape=e.outShape,g=Ye({inputs:{x:T},backend:s}),g.shape=e.outShape,x.push(T)}else{const y=e.outHeight*e.outWidth,$=et({inputs:{x:n},backend:s,attrs:{shape:p?[e.batchSize,y,e.inChannels]:[e.batchSize,e.inChannels,y]}}),v=et({inputs:{x:t},backend:s,attrs:{shape:[1,e.inChannels,e.outChannels]}}),N=hl({a:p?$:v,b:p?v:$,transposeA:!p,transposeB:m,backend:s,bias:o,activation:a,preluActivationWeights:r,leakyreluAlpha:i});g=et({inputs:{x:N},backend:s,attrs:{shape:e.outShape}}),x.push($),x.push(v),x.push(N)}for(const y of x)s.disposeIntermediateTensorInfo(y);return g}function jw({x:n,filter:t,convInfo:e,backend:s,bias:o=null,preluActivationWeights:r=null,leakyreluAlpha:i=0,activation:a=null}){const{filterWidth:l,filterHeight:c,inChannels:u,outWidth:h,outHeight:d,dataFormat:p}=e,f=p==="channelsLast",m=l*c*u,g=d*h,x=[e.batchSize,m,g],b=!0,w=!1,y=[];if(r!=null){const W=dl(r.shape,f);W!=null&&(r=et({inputs:{x:r},backend:s,attrs:{shape:W}}),y.push(r))}if(o!=null){const W=dl(o.shape,f);W!=null&&(o=et({inputs:{x:o},backend:s,attrs:{shape:W}}),y.push(o))}const $=et({inputs:{x:t},backend:s,attrs:{shape:[1,m,U(t.shape)/m]}});y.push($);const v=new DW(x,e),N=[n.shape,[e.padInfo.top,e.padInfo.left],[e.strideHeight,e.strideWidth],[e.dilationHeight,e.dilationWidth],[e.inChannels],[e.filterWidth*e.inChannels],[e.outWidth]],T=s.runWebGLProgram(v,[n],"float32",N),k=et({inputs:{x:T},backend:s,attrs:{shape:x}});y.push(T),y.push(k);const S=o!=null,C=r!=null,D=a==="leakyrelu",O=a?Hr(a,!0):null,P=new Mw(f?k.shape:$.shape,f?$.shape:k.shape,f?[e.batchSize,g,e.outChannels]:[e.batchSize,e.outChannels,g],b,w,S,O,C,D),M=f?[k,$]:[$,k];if(o&&M.push(o),C&&M.push(r),D){const W=s.makeTensorInfo([],"float32",as(i,"float32"));M.push(W),y.push(W)}const V=s.runWebGLProgram(P,M,"float32"),H=et({inputs:{x:V},backend:s,attrs:{shape:e.outShape}});y.push(V);for(const W of y)s.disposeIntermediateTensorInfo(W);return H}function FW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dataFormat:l,dilations:c,dimRoundingMode:u}=s,h=us(l),d=Ce(o.shape,r.shape,i,c,a,u,!1,h);let p;if(d.filterHeight===1&&d.filterWidth===1&&d.dilationHeight===1&&d.dilationWidth===1&&d.strideHeight===1&&d.strideWidth===1&&(d.padInfo.type==="SAME"||d.padInfo.type==="VALID"))p=qw({x:o,filter:r,convInfo:d,backend:e});else if(d.strideWidth<=2&&h==="channelsLast"&&B().getBool("WEBGL_EXP_CONV")){const m=new Hw(d),g=[[d.padInfo.top,d.padInfo.left],[d.strideHeight,d.strideWidth],[d.dilationHeight,d.dilationWidth],[d.inHeight,d.inWidth]];p=e.runWebGLProgram(m,[o,r],"float32",g)}else if(B().getBool("WEBGL_CONV_IM2COL"))p=jw({x:o,filter:r,convInfo:d,backend:e});else{const m=new Gw(d);p=e.runWebGLProgram(m,[o,r],"float32")}const f=et({inputs:{x:p},backend:e,attrs:{shape:d.outShape}});return e.disposeIntermediateTensorInfo(p),f}const _W={kernelName:kl,backendName:"webgl",kernelFunc:FW};class OW{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,s=t.strideWidth,o=t.padInfo.top,r=t.padInfo.left,i=t.dataFormat==="channelsLast";this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int d2 = coords.w;

        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yR = 0; yR < ${t.outHeight}; yR++) {
            int xR = wR + yR * ${e} - ${o};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${t.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              ${i?`float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);`:`float dyValue = getDy(b, d2, yR, yC);
              float xValue = getX(b, d1, xR, xC);
              dotProd += (xValue * dyValue);`}
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class LW{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=t.dataFormat==="channelsLast",a=e-1-t.padInfo.top,l=s-1-t.padInfo.left,c=i?1:2,u=i?2:3,h=i?3:1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${l});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[${h}];

        ivec2 dyCorner = ivec2(coords[${c}], coords[${u}]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            for (int d2 = 0; d2 < ${t.outChannels}; d2++) {

              if (${i}) {
                float xValue = getDy(batch, idyR, idyC, d2);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              } else {
                float xValue = getDy(batch, d2, idyR, idyC);
                float wValue = getW(wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }

            }
          }
        }
        setOutput(dotProd);
      }
    `}}class MW{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideDepth,s=t.strideHeight,o=t.strideWidth,r=t.padInfo.front,i=t.padInfo.top,a=t.padInfo.left;this.userCode=`
      void main() {
        ivec5 coords = getOutputCoords();
        int wF = coords.x;
        int wR = coords.y;
        int wC = coords.z;
        int d1 = coords.w;
        int d2 = coords.u;

        float dotProd = 0.0;

        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yF = 0; yF < ${t.outDepth}; yF++) {
            int xF = wF + yF * ${e} - ${r};

            if (xF < 0 || xF >= ${t.inDepth}) {
              continue;
            }

            for (int yR = 0; yR < ${t.outHeight}; yR++) {
              int xR = wR + yR * ${s} - ${i};

              if (xR < 0 || xR >= ${t.inHeight}) {
                continue;
              }

              for (int yC = 0; yC < ${t.outWidth}; yC++) {
                int xC = wC + yC * ${o} - ${a};

                if (xC < 0 || xC >= ${t.inWidth}) {
                  continue;
                }

                float dyValue = getDy(b, yF, yR, yC, d2);
                float xValue = getX(b, xF, xR, xC, d1);
                dotProd += (xValue * dyValue);
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class PW{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterDepth,s=t.filterHeight,o=t.filterWidth,r=t.strideDepth,i=t.strideHeight,a=t.strideWidth,l=e-1-t.padInfo.front,c=s-1-t.padInfo.top,u=o-1-t.padInfo.left;this.userCode=`
      const ivec3 pads = ivec3(${l}, ${c}, ${u});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.u;


        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyFCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        float dotProd = 0.0;
        for (int wF = 0; wF < ${e}; wF++) {
          float dyF = float(dyFCorner + wF) / ${r}.0;

          if (dyF < 0.0 || dyF >= ${t.outDepth}.0 || fract(dyF) > 0.0) {
            continue;
          }
          int idyF = int(dyF);

          int wFPerm = ${e} - 1 - wF;

          for (int wR = 0; wR < ${s}; wR++) {
            float dyR = float(dyRCorner + wR) / ${i}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
              fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            int wRPerm = ${s} - 1 - wR;

            for (int wC = 0; wC < ${o}; wC++) {
              float dyC = float(dyCCorner + wC) / ${a}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              int wCPerm = ${o} - 1 - wC;

              for (int d2 = 0; d2 < ${t.outChannels}; d2++) {
                float xValue = getDy(batch, idyF, idyR, idyC, d2);
                float wValue = getW(wFPerm, wRPerm, wCPerm, d1, d2);
                dotProd += xValue * wValue;
              }
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function zW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,dataFormat:l,dimRoundingMode:c,filterShape:u}=s,h=us(l),d=Ce(o.shape,u,i,1,a,c,!1,h),p=new OW(d);return e.runWebGLProgram(p,[o,r],"float32")}const BW={kernelName:hh,backendName:"webgl",kernelFunc:zW};class VW{constructor(t){this.variableNames=["dy","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"strides",type:"vec2"}],this.outputShape=t.inShape,this.enableShapeUniforms=Re(this.outputShape.length);const e=t.filterHeight,s=t.filterWidth,o=e-1-t.padInfo.top,r=s-1-t.padInfo.left;this.userCode=`
      const ivec2 pads = ivec2(${o}, ${r});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];

        ivec2 dyCorner = ivec2(coords[1], coords[2]) - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        vec4 result = vec4(0.);
        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / strides[0];
          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);
          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            int wCPerm = ${s} - 1 - wC;

            float dyC = float(dyCCorner + wC) / strides[1];
            bool idyCVal = (dyC >= 0.0) && (dyC < ${t.outWidth}.0)
              && (fract(dyC) == 0.0);
            int idyC = int(dyC);

            float dyC2 = float(dyCCorner + wC + 1) / strides[1];
            bool idyCVal2 = (dyC2 >= 0.0) && (dyC2 < ${t.outWidth}.0)
              && (fract(dyC2) == 0.0);
            int idyC2 = int(dyC2);

            if (idyCVal && idyCVal2) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec4 dySample2 = (idyC / 2 == idyC2 / 2) ?
                  dySample : getDy(batch, idyR, idyC2, d2);

                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));

                dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample2.xy : dySample2.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC, d2);
                vec2 dyValue = mod(float(idyC), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.xy += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            } else if (idyCVal2) {
              for (int d2 = 0; d2 < ${t.outChannels}; d2 += 2) {
                vec4 wValue = getW(wRPerm, wCPerm, d1, d2);
                vec4 dySample = getDy(batch, idyR, idyC2, d2);
                vec2 dyValue = mod(float(idyC2), 2.) == 0. ?
                  dySample.xy : dySample.zw;
                result.zw += vec2(dot(dyValue, wValue.xy),
                  dot(dyValue, wValue.zw));
              }
            }
          }
        }
        setOutput(result);
      }
    `}}function WW(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{inputShape:i,strides:a,pad:l,dataFormat:c,dimRoundingMode:u}=s,h=us(c),d=Ce(i,r.shape,a,1,l,u,!1,h);if(B().getBool("WEBGL_PACK_CONV2DTRANSPOSE")&&h==="channelsLast"){const p=[[d.strideHeight,d.strideWidth]],f=new VW(d);return e.runWebGLProgram(f,[o,r],"float32",p)}else{const p=new LW(d);return e.runWebGLProgram(p,[o,r],"float32")}}const UW={kernelName:Sl,backendName:"webgl",kernelFunc:WW};function GW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s,c=Ms(o.shape,r.shape,i,l,a),u=new AW(c);return e.runWebGLProgram(u,[o,r],"float32")}const HW={kernelName:Nl,backendName:"webgl",kernelFunc:GW};function qW(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,pad:a,filterShape:l}=s,c=Ms(o.shape,l,i,1,a),u=new MW(c);return e.runWebGLProgram(u,[o,r],"float32")}const jW={kernelName:dh,backendName:"webgl",kernelFunc:qW};function KW(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{pad:i,strides:a,inputShape:l}=s,c=Ms(l,r.shape,a,1,i),u=new PW(c);return e.runWebGLProgram(u,[o,r],"float32")}const XW={kernelName:ph,backendName:"webgl",kernelFunc:KW};const YW=pr+`
  return cos(x);
`,ZW=`
  vec4 result = cos(x);
  bvec4 isNaN = isnan(x);
  ${Do}
  return result;
`,JW=St({opSnippet:YW,packedOpSnippet:ZW}),QW={kernelName:ri,backendName:"webgl",kernelFunc:JW};const t4=`
  float e2x = exp(-x);
  return (e2x + 1.0 / e2x) / 2.0;
`,e4=St({opSnippet:t4}),n4={kernelName:ii,backendName:"webgl",kernelFunc:e4};class s4{constructor(t,e,s,o,r){this.variableNames=["Image","Boxes","BoxInd"],this.outputShape=[];const[i,a,l,c]=t,[u]=e,[h,d]=s;this.outputShape=[u,h,d,c];const p=o==="bilinear"?1:0,[f,m]=[`${a-1}.0`,`${l-1}.0`],[g,x,b]=h>1?[`${(a-1)/(h-1)}`,"(y2-y1) * height_ratio",`y1*${f} + float(y)*(height_scale)`]:["0.0","0.0",`0.5 * (y1+y2) * ${f}`],[w,y,$]=d>1?[`${(l-1)/(d-1)}`,"(x2-x1) * width_ratio",`x1*${m} + float(x)*(width_scale)`]:["0.0","0.0",`0.5 * (x1+x2) * ${m}`];this.userCode=`
      const float height_ratio = float(${g});
      const float width_ratio = float(${w});
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int y = coords[1];
        int x = coords[2];
        int d = coords[3];

        // get box vals
        float y1 = getBoxes(b,0);
        float x1 = getBoxes(b,1);
        float y2 = getBoxes(b,2);
        float x2 = getBoxes(b,3);

        // get image in batch index
        int bInd = round(getBoxInd(b));
        if(bInd < 0 || bInd >= ${i}) {
          return;
        }

        float height_scale = ${x};
        float width_scale = ${y};

        float in_y = ${b};
        if( in_y < 0.0 || in_y > ${f} ) {
          setOutput(float(${r}));
          return;
        }
        float in_x = ${$};
        if( in_x < 0.0 || in_x > ${m} ) {
          setOutput(float(${r}));
          return;
        }

        vec2 sourceFracIndexCR = vec2(in_x,in_y);
        if(${p} == 1) {
          // Compute the four integer indices.
          ivec2 sourceFloorCR = ivec2(sourceFracIndexCR);
          ivec2 sourceCeilCR = ivec2(ceil(sourceFracIndexCR));

          float topLeft = getImage(b, sourceFloorCR.y, sourceFloorCR.x, d);
          float bottomLeft = getImage(b, sourceCeilCR.y, sourceFloorCR.x, d);
          float topRight = getImage(b, sourceFloorCR.y, sourceCeilCR.x, d);
          float bottomRight = getImage(b, sourceCeilCR.y, sourceCeilCR.x, d);

          vec2 fracCR = sourceFracIndexCR - vec2(sourceFloorCR);

          float top = topLeft + (topRight - topLeft) * fracCR.x;
          float bottom = bottomLeft + (bottomRight - bottomLeft) * fracCR.x;
          float newValue = top + (bottom - top) * fracCR.y;
          setOutput(newValue);
        } else {
          // Compute the coordinators of nearest neighbor point.
          ivec2 sourceNearestCR = ivec2(floor(
            sourceFracIndexCR + vec2(0.5,0.5)));
          float newValue = getImage(b, sourceNearestCR.y, sourceNearestCR.x, d);
          setOutput(newValue);
        }
      }
    `}}const o4=n=>{const{inputs:t,backend:e,attrs:s}=n,{image:o,boxes:r,boxInd:i}=t,{cropSize:a,method:l,extrapolationValue:c}=s,u=new s4(o.shape,r.shape,a,l,c);return e.runWebGLProgram(u,[o,r,i],"float32")},r4={kernelName:mh,backendName:"webgl",kernelFunc:o4};var jr;(function(n){n.Prod="*",n.Sum="+"})(jr||(jr={}));class _m{constructor(t,e,s,o){this.op=t,this.outputShape=e,this.variableNames=["x"],this.customUniforms=[{name:"index",type:"float"}];const r=this.outputShape.length,i=this.op===jr.Prod?"1.0":"0.0",a=s?i:`getX(${Om(r,"coords",this.op)})`,l=this.outputShape[this.outputShape.length-1];let c="",u="";s?(c=o?`end != ${l-1}`:"end != 0",u=o?"end + 1":"end - 1"):(c=o?`end + pow2 < ${l}`:"end >= pow2",u=o?"end + pow2":"end - pow2"),this.userCode=`
      void main() {
        ${Ot(r)} coords = getOutputCoords();
        int end = ${Lm(r,"coords",this.op)};
        float val = ${a};
        int pow2 = int(pow(2.0, index));
        if (${c}) {
          int idx = ${u};
          ${Lm(r,"coords",this.op)} = idx;
          val ${this.op}= getX(${Om(r,"coords",this.op)});
        }
        setOutput(val);
      }
    `}}function Om(n,t,e){if(n===1)return`${t}`;if(n===2)return`${t}.x, ${t}.y`;if(n===3)return`${t}.x, ${t}.y, ${t}.z`;if(n===4)return`${t}.x, ${t}.y, ${t}.z, ${t}.w`;throw new Error(`Cumulative ${e} for rank ${n} is not yet supported`)}function Lm(n,t,e){if(n===1)return`${t}`;if(n===2)return`${t}.y`;if(n===3)return`${t}.z`;if(n===4)return`${t}.w`;throw new Error(`Cumulative ${e} for rank ${n} is not yet supported`)}function Kw(n,t,e,s,o,r){const i=t.shape.length,a=Xt([s],i);let l=t;a!=null&&(l=_e({inputs:{x:t},backend:e,attrs:{perm:a}}));const c=ne(1,i)[0];if(c!==i-1)throw new Error(`WebGL cumprod shader expects an inner-most axis=${t.shape.length-1} but got axis=${s}`);const u=l.shape[c];let h=Ye({inputs:{x:l},backend:e});for(let d=0;d<=Math.ceil(Math.log2(u))-1;d++){const p=new _m(n,l.shape,!1,r),f=[[d]],m=h;h=e.runWebGLProgram(p,[h],h.dtype,f),e.disposeIntermediateTensorInfo(m)}if(o){const d=new _m(n,l.shape,o,r),p=h;h=e.runWebGLProgram(d,[h],h.dtype),e.disposeIntermediateTensorInfo(p)}if(a!=null){const d=Ps(a),p=_e({inputs:{x:h},backend:e,attrs:{perm:d}});return e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(l),p}return h}function i4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;return Kw(jr.Prod,o,e,r,i,a)}const a4={kernelName:fh,backendName:"webgl",kernelFunc:i4};function l4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,exclusive:i,reverse:a}=s;return Kw(jr.Sum,o,e,r,i,a)}const c4={kernelName:Tl,backendName:"webgl",kernelFunc:l4};function u4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,weights:r}=t,{size:i,binaryOutput:a}=s;if(o.shape.length===1){const l=e.readSync(o.dataId),c=e.readSync(r.dataId),u=Tw(l,c,r.dtype,r.shape,i);return e.makeTensorInfo([i],r.dtype,u)}else if(o.shape.length===2){const l=e.bufferSync(o),c=e.bufferSync(r),u=pz(l,c,i,a);return e.makeTensorInfo(u.shape,r.dtype,u.values)}throw new Error(`Error in denseBincount: input must be at most rank 2, but got rank${o.shape.length}.`)}const h4={kernelName:gh,backendName:"webgl",kernelFunc:u4};class d4{constructor(t,e,s){this.variableNames=["x"],this.outputShape=[],this.outputShape=t,this.blockSize=e,this.dataFormat=s,this.userCode=`
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int h = ${this.getHeightCoordString()};
      int w = ${this.getWidthCoordString()};
      int d = ${this.getDepthCoordString()};

      int in_h = h / ${e};
      int offset_h = imod(h, ${e});
      int in_w = w / ${e};
      int offset_w = imod(w, ${e});
      int offset_d = (offset_h * ${e} + offset_w) *
        ${this.getOutputDepthSize()};
      int in_d = d + offset_d;

      float result = ${this.getInputSamplingString()};
      setOutput(result);
    }
  `}getHeightCoordString(){return this.dataFormat==="NHWC"?"coords[1]":"coords[2]"}getWidthCoordString(){return this.dataFormat==="NHWC"?"coords[2]":"coords[3]"}getDepthCoordString(){return this.dataFormat==="NHWC"?"coords[3]":"coords[1]"}getOutputDepthSize(){return this.dataFormat==="NHWC"?this.outputShape[3]:this.outputShape[1]}getInputSamplingString(){return this.dataFormat==="NHWC"?"getX(b, in_h, in_w, in_d)":"getX(b, in_d, in_h, in_w)"}}function p4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockSize:r,dataFormat:i}=s,a=o.shape[0],l=i==="NHWC"?o.shape[1]:o.shape[2],c=i==="NHWC"?o.shape[2]:o.shape[3],u=i==="NHWC"?o.shape[3]:o.shape[1],h=l*r,d=c*r,p=u/(r*r),f=i==="NHWC"?[a,h,d,p]:[a,p,h,d],m=new d4(f,r,i);return e.runWebGLProgram(m,[o],o.dtype)}const f4={kernelName:xh,backendName:"webgl",kernelFunc:p4};class Xw{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Re(this.outputShape.length);const i=t.filterHeight,a=t.filterWidth,l=t.outChannels/t.inChannels;let c="",u="";s&&(o?c=`float activation(float a) {
          float b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?c=`float activation(float a) {
          float b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:c=`
          float activation(float x) {
            ${s}
          }
        `,u="result = activation(result);");const h=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${c}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${l};
        int q = d2 - d1 * ${l};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        // TO DO(dsmilkov): Flatten the two for loops and vec4 the operations.
        for (int wR = 0; wR < ${i}; wR++) {
          int xR = xRCorner + wR * dilations[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < ${a}; wC++) {
            int xC = xCCorner + wC * dilations[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            float xVal = getX(batch, xR, xC, d1);
            float wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }

        float result = dotProd;
        ${h}
        ${u}
        setOutput(result);
      }
    `}}class Yw{constructor(t,e=!1,s=null,o=!1,r=!1){this.variableNames=["x","W"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"pads",type:"ivec2"},{name:"strides",type:"ivec2"},{name:"dilations",type:"ivec2"},{name:"inDims",type:"ivec2"}],this.outputShape=t.outShape,this.enableShapeUniforms=Re(this.outputShape.length);const i=t.outChannels/t.inChannels,a=t.padInfo.left,l=t.strideWidth,c=t.dilationWidth,u=t.filterHeight,h=t.filterWidth,d=h;let p=`
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;for(let x=0;x<h;x++)p+=`
          vec4 xTexelC${x*2};
          int xTexelC${x*2}Ready;
          vec4 xTexelC${x*2+1};
          int xTexelC${x*2+1}Ready;
          vec4 xC${x};`;p+=`
    for (int r = 0; r < ${u}; r++) {
      `;for(let x=0;x<h;x++)p+=`
          xTexelC${x*2} = vec4(0.0);
          xTexelC${x*2}Ready = 0;
          xTexelC${x*2+1} = vec4(0.0);
          xTexelC${x*2+1}Ready = 0;
          xC${x} = vec4(0.0);`;p+=`
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;for(let x=0;x<(d+1)/2;x++){const b=x*2;if(p+=`
          xC = xCCorner + ${b*c};
          `,l===1){if(b<h&&(a%2===1?(p+=`
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }
              `,c===1&&b>0?p+=`
                xC${b} = vec4(xTexelC${b-2}.zw, xTexelC${b}.xy);
                `:p+=`
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${b} = vec4(previous.zw, xTexelC${b}.xy);
                  } else {
                    xC${b} = vec4(0.0, 0.0, xTexelC${b}.xy);
                  }
                  `):p+=`
                if (xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xC${b} = xTexelC${b};
                `,b+1<h)){const w=a%2===0?fl(c):c;c%2===0&&a%2===1||c%2!==0&&a%2!==1?(p+=`
                  xCOffset = xC + imod(pads[1], 2) + ${w};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                    xTexelC${b+1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${b+1}.zw = vec2(0.0);
                    }
                    xTexelC${b+1}Ready = 1;
                  }
                  `,c>1?p+=`
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1]) {
                     previous = getX(batch, xR, xCOffset, d1);
                     xC${b+1} = vec4(previous.zw, xTexelC${b+1}.xy);
                    } else {
                     xC${b+1} = vec4(0.0, 0.0, xTexelC${b+1}.xy);
                    }
                    `:p+=`
                    xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.xy);
                    `):w===1?p+=`
                    xC${b+1} = xTexelC${b};
                    `:p+=`
                    xCOffset = xC + ${w};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                      xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${b+1}.zw = vec2(0.0);
                      }
                      xTexelC${b+1}Ready = 1;
                    }

                    xC${b+1} = xTexelC${b+1};
                    `}}else b<h&&(a%2===1?(p+=`
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.0);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
              `,b+1<h&&(p+=`
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${b+1} = vec4(xTexelC${b+1}.xy, final.xy);
                `)):(p+=`
                if(xC >= 0 && xC < inDims[1] && xTexelC${b}Ready == 0) {
                  xTexelC${b} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${b}.zw = vec2(0.0);
                  }
                  xTexelC${b}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${b+1}Ready == 0) {
                  xTexelC${b+1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${b+1}.zw = vec2(0.);
                  }
                  xTexelC${b+1}Ready = 1;
                }

                xC${b} = vec4(
                  xTexelC${b}.xy, xTexelC${b+1}.xy);
              `,b+1<h&&(p+=`
                  xC${b+1} = vec4(xTexelC${b}.zw, xTexelC${b+1}.zw);
                `)));b<h&&(p+=`
            wTexel = getW(r, ${b}, d1, q);
            dotProd += xC${b} * vec4(wTexel.xz, wTexel.xz);
          `,b+1<h&&(p+=`
              wTexel = getW(r, ${b+1}, d1, q);
              dotProd += xC${b+1} * vec4(wTexel.xz, wTexel.xz);
            `))}p+=`
    }
  `,p+=`
      }
    `;let f="",m="";s&&(o?f=`vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${s}
        }`:r?f=`vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${s}
        }`:f=`vec4 activation(vec4 x) {
          ${s}
        }`,m="result = activation(result);");const g=e?"result += getBiasAtOutCoords();":"";e&&this.variableNames.push("bias"),o&&this.variableNames.push("preluActivationWeights"),r&&this.variableNames.push("leakyreluAlpha"),this.userCode=`
      ${f}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${i};
        int q = d2 - d1 * ${i};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${p}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${g}
        ${m}
        setOutput(result);
      }
    `}}function m4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l,dimRoundingMode:c}=s;let u=l;u==null&&(u=[1,1]),I(Te(i,u),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${i} and dilations '${u}'`);const h=Ce(o.shape,r.shape,i,u,a,c,!0);let d;B().getBool("WEBGL_PACK_DEPTHWISECONV")&&h.strideWidth<=2&&h.outChannels/h.inChannels===1?d=new Yw(h):d=new Xw(h);const p=[[h.padInfo.top,h.padInfo.left],[h.strideHeight,h.strideWidth],[h.dilationHeight,h.dilationWidth],[h.inHeight,h.inWidth]];return e.runWebGLProgram(d,[o,r],"float32",p)}const g4={kernelName:El,backendName:"webgl",kernelFunc:m4};class x4{constructor(t){this.variableNames=["x","dy"],this.outputShape=t.filterShape;const e=t.strideHeight,s=t.strideWidth,o=t.padInfo.top,r=t.padInfo.left,i=t.outChannels/t.inChannels;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int wR = coords.x;
        int wC = coords.y;
        int d1 = coords.z;
        int dm = coords.w;
        int d2 = d1 * ${i} + dm;

        float dotProd = 0.0;

        // TO DO: Vec4 over the batch size
        for (int b = 0; b < ${t.batchSize}; b++) {
          for (int yR = 0; yR < ${t.outHeight}; yR++) {
            int xR = wR + yR * ${e} - ${o};

            if (xR < 0 || xR >= ${t.inHeight}) {
              continue;
            }

            for (int yC = 0; yC < ${t.outWidth}; yC++) {
              int xC = wC + yC * ${s} - ${r};

              if (xC < 0 || xC >= ${t.inWidth}) {
                continue;
              }

              float dyValue = getDy(b, yR, yC, d2);
              float xValue = getX(b, xR, xC, d1);
              dotProd += (xValue * dyValue);
            }
          }
        }
        setOutput(dotProd);
      }
    `}}class b4{constructor(t){this.variableNames=["dy","W"],this.outputShape=t.inShape;const e=t.filterHeight,s=t.filterWidth,o=t.strideHeight,r=t.strideWidth,i=e-1-t.padInfo.top,a=s-1-t.padInfo.left,l=t.outChannels/t.inChannels;this.userCode=`
      const ivec2 pads = ivec2(${i}, ${a});

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords[0];
        int d1 = coords[3];
        ivec2 dyCorner = coords.yz - pads;
        int dyRCorner = dyCorner.x;
        int dyCCorner = dyCorner.y;

        float dotProd = 0.0;

        for (int wR = 0; wR < ${e}; wR++) {
          float dyR = float(dyRCorner + wR) / ${o}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          int wRPerm = ${e} - 1 - wR;

          for (int wC = 0; wC < ${s}; wC++) {
            float dyC = float(dyCCorner + wC) / ${r}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            int wCPerm = ${s} - 1 - wC;

            // TO DO: Vec4 over the channelMul
            for (int dm = 0; dm < ${l}; dm++) {
              int d2 = d1 * ${l} + dm;
              float xValue = getDy(batch, idyR, idyC, d2);
              float wValue = getW(wRPerm, wCPerm, d1, dm);
              dotProd += xValue * wValue;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function y4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,dy:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,filterShape:u}=s,h=Ce(o.shape,u,i,a,l,c,!0),d=new x4(h);return e.runWebGLProgram(d,[o,r],"float32")}const w4={kernelName:bh,backendName:"webgl",kernelFunc:y4};function $4(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,filter:r}=t,{strides:i,dilations:a,pad:l,dimRoundingMode:c,inputShape:u}=s,h=Ce(u,r.shape,i,a,l,c,!0),d=new b4(h);return e.runWebGLProgram(d,[o,r],"float32")}const C4={kernelName:yh,backendName:"webgl",kernelFunc:$4};class I4{constructor(t){this.variableNames=["X"],this.outputShape=[t,t],this.userCode=`
      void main() {
          ivec2 coords = getOutputCoords();
          float val = coords[0] == coords[1] ? getX(coords[0]) : 0.0;
          setOutput(val);
      }
    `}}function v4(n){const{inputs:t,backend:e}=n,{x:s}=t,o=[...s.shape,...s.shape],r=U(s.shape),i=et({inputs:{x:s},backend:e,attrs:{shape:[r]}}),a=new I4(r),l=e.runWebGLProgram(a,[i],i.dtype),c=et({inputs:{x:l},backend:e,attrs:{shape:o}});return e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(l),c}const k4={kernelName:eg,backendName:"webgl",kernelFunc:v4};class S4{constructor(t){this.variableNames=["x","W"],this.outputShape=t.outShape;const{inHeight:e,inWidth:s,padInfo:o,strideHeight:r,strideWidth:i,filterHeight:a,filterWidth:l,dilationHeight:c,dilationWidth:u}=t,{top:h,left:d}=o;this.userCode=`
      const ivec2 strides = ivec2(${r}, ${i});
      const ivec2 pads = ivec2(${h}, ${d});
      const float neg_infinity = -3.4e38;

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        int d1 = coords.w;
        ivec2 outTopLeftCorner =
            coords.yz * strides - pads;
        int hBeg = outTopLeftCorner.x;
        int wBeg = outTopLeftCorner.y;

        float curVal = neg_infinity;
        for (int h = 0; h < ${a}; h++) {
          int hIn = hBeg + h * ${c};

          if (hIn >= 0 && hIn < ${e}) {
            for (int w = 0; w < ${l}; w++) {
              int wIn = wBeg + w * ${u};

              if (wIn >= 0 && wIn < ${s}) {
                float xVal = getX(batch, hIn, wIn, d1);
                float wVal = getW(h, w, d1);

                float val = xVal + wVal;
                if (val > curVal) {
                  curVal = val;
                }
              }
            }
          }
        }

        float result = curVal;
        setOutput(result);
      }
    `}}function N4(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r}=t,{strides:i,pad:a,dilations:l}=s,c=Hi(o.shape,r.shape,i,a,"NHWC",l);let u;const h=new S4(c);u=e.runWebGLProgram(h,[o,r],"float32");const d=et({inputs:{x:u},backend:e,attrs:{shape:c.outShape}});return e.disposeIntermediateTensorInfo(u),d}const T4={kernelName:Rl,backendName:"webgl",kernelFunc:N4};function E4(n){const{inputs:t,backend:e,attrs:s}=n,{equation:o}=s,r=t,{allDims:i,summedDims:a,idDims:l}=op(o,r.length);ip(i.length,l,r);const{path:c,steps:u}=ap(a,l),h=u.length;let d=null,p=i.length;const f=[];for(let m=0;m<h;++m){for(const g of u[m]){const{permutationIndices:x,expandDims:b}=rp(p,l[g]);let w;lp(x)?w=r[g]:(w=_e({inputs:{x:r[g]},backend:e,attrs:{perm:x}}),f.push(w));const y=w.shape.slice();for(let $=0;$<b.length;++$)y.splice(b[$],0,1);Tt(w.shape,y)||(w=et({inputs:{x:w},backend:e,attrs:{shape:y}}),f.push(w)),d===null?d=w:(d=nf({inputs:{a:w,b:d},backend:e}),f.push(d))}m<h-1&&(c[m]>=0&&(d=Kc({inputs:{x:d},backend:e,attrs:{axis:c[m]-(i.length-p),keepDims:!1}}),f.push(d)),p--)}for(const m of f)m!==d&&e.disposeIntermediateTensorInfo(m);return d}const R4={kernelName:wh,backendName:"webgl",kernelFunc:E4};const A4="return (x >= 0.0) ? x : (exp(x) - 1.0);",D4=`
  vec4 result;

  result.r = (x.r >= 0.0) ? x.r : (exp(x.r) - 1.0);
  result.g = (x.g >= 0.0) ? x.g : (exp(x.g) - 1.0);
  result.b = (x.b >= 0.0) ? x.b : (exp(x.b) - 1.0);
  result.a = (x.a >= 0.0) ? x.a : (exp(x.a) - 1.0);

  return result;
`,F4=St({opSnippet:A4,packedOpSnippet:D4}),_4={kernelName:li,backendName:"webgl",kernelFunc:F4};const O4="return (b >= 0.0) ? a : a * (b + 1.0);",L4=`
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`,M4=n=>{const{inputs:t,backend:e}=n,{dy:s,y:o}=t,r=B().getBool("WEBGL_PACK_BINARY_OPERATIONS")?new dr(L4,s.shape,o.shape):new Io(O4,s.shape,o.shape);return e.runWebGLProgram(r,[s,o],s.dtype)},P4={kernelName:$h,backendName:"webgl",kernelFunc:M4};const z4=`
  return vec4(equal(a, b));
`,B4="return float(a == b);",V4=ve({opSnippet:B4,packedOpSnippet:z4,dtype:"bool",cpuKernelImpl:bz}),W4={kernelName:Al,backendName:"webgl",kernelFunc:V4};const U4=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  float p = ${Zd};
  float a1 = ${Jd};
  float a2 = ${Qd};
  float a3 = ${tp};
  float a4 = ${ep};
  float a5 = ${np};

  float sign = sign(x);
  x = abs(x);
  float t = 1.0 / (1.0 + p * x);
  return sign * (1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x));
`,G4=St({opSnippet:U4}),H4={kernelName:ci,backendName:"webgl",kernelFunc:G4};const q4=pr+`
  return exp(x);
`,j4=`
  vec4 result = exp(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Zw=St({opSnippet:q4,packedOpSnippet:j4,cpuKernelImpl:yz,dtype:"float32"}),K4={kernelName:ui,backendName:"webgl",kernelFunc:Zw};function Xu(n){const{inputs:t,attrs:e,backend:s}=n,{dim:o}=e,{input:r}=t,i=r.shape.length,a=r.shape.slice();let l=o;return o<0&&(I(-(i+1)<=o,()=>`Axis must be in the interval [${-(i+1)}, ${i}]`),l=i+o+1),a.splice(l,0,1),et({inputs:{x:r},backend:s,attrs:{shape:a}})}const X4={kernelName:Dl,backendName:"webgl",kernelFunc:Xu};const Mm="return exp(x) - 1.0;",Y4=St({opSnippet:Mm,packedOpSnippet:Mm,cpuKernelImpl:wz}),Z4={kernelName:hi,backendName:"webgl",kernelFunc:Y4};class Pm{constructor(t,e,s){this.variableNames=["real","imag"];const o=e[1];this.outputShape=e;const r=s?`2.0 * ${Math.PI}`:`-2.0 * ${Math.PI}`,i=s?`${o}.0`:"1.0";let a;if(t==="real")a="return real * expR - imag * expI;";else if(t==="imag")a="return real * expI + imag * expR;";else throw new Error(`FFT component must be either "real" or "imag", got ${t}.`);this.userCode=`
      const float exponentMultiplier = ${r};

      float unaryOpComplex(float real, float expR, float imag, float expI) {
        ${a}
      }

      float mulMatDFT(int batch, int index) {
        float indexRatio = float(index) / float(${o});
        float exponentMultiplierTimesIndexRatio =
            exponentMultiplier * indexRatio;

        float result = 0.0;

        for (int i = 0; i < ${o}; i++) {
          // x = (-2|2 * PI / N) * index * i;
          float x = exponentMultiplierTimesIndexRatio * float(i);
          float expR = cos(x);
          float expI = sin(x);
          float real = getReal(batch, i);
          float imag = getImag(batch, i);

          result +=
              unaryOpComplex(real, expR, imag, expI) / ${i};
        }

        return result;
      }

      void main() {
        ivec2 coords = getOutputCoords();
        setOutput(mulMatDFT(coords[0], coords[1]));
      }
    `}}function Jw(n,t,e){const s=e.texData.get(n.dataId),o=U(n.shape),r=n.shape[n.shape.length-1],i=o/r,a=et({inputs:{x:n},backend:e,attrs:{shape:[i,r]}}),l=a.shape,c=new Pm("real",l,t),u=new Pm("imag",l,t),h=[{dataId:s.complexTensorInfos.real.dataId,dtype:s.complexTensorInfos.real.dtype,shape:l},{dataId:s.complexTensorInfos.imag.dataId,dtype:s.complexTensorInfos.imag.dtype,shape:l}],d=e.runWebGLProgram(c,h,"float32"),p=e.runWebGLProgram(u,h,"float32"),f=Gs({inputs:{real:d,imag:p},backend:e});e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p);const m=et({inputs:{x:f},backend:e,attrs:{shape:n.shape}});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(f),m}function J4(n){const{inputs:t,backend:e}=n,{input:s}=t;return Jw(s,!1,e)}const Q4={kernelName:Ch,backendName:"webgl",kernelFunc:J4};class tU{constructor(t,e){this.outputShape=[],this.customUniforms=[{name:"value",type:"float"}],this.variableNames=["x"],this.outputShape=t,this.userCode=`
      void main() {
        // Input can be obtained from uniform value.
        setOutput(value);
      }
    `}}function da(n){const{backend:t,attrs:e}=n,{shape:s,value:o}=e;let{dtype:r}=e;if(r=r||vo(o),r==="string"){const i=Jt(r,U(s));return i.fill(o),t.makeTensorInfo(s,r,i)}else{const i=new tU(s,o),a=[[o]];return t.runWebGLProgram(i,[],r,a)}}const eU={kernelName:Ih,backendName:"webgl",kernelFunc:da};class nU{constructor(t){this.variableNames=["Image"],this.outputShape=[];const e=t[2];this.outputShape=t,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];

          int coordX = ${e} - x - 1;
          float outputValue;
          if(coordX >= 0 && coordX < ${e}) {
            outputValue = getImage(coords[0], coords[1], coordX, coords[3]);
          } else {
            outputValue = getImage(coords[0], coords[1], coords[2], coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const sU={kernelName:vh,backendName:"webgl",kernelFunc:({inputs:n,backend:t})=>{const{image:e}=n,s=t,o=new nU(e.shape);return s.runWebGLProgram(o,[e],e.dtype)}};const zm="return floor(x);",oU=St({opSnippet:zm,packedOpSnippet:zm,cpuKernelImpl:$z}),rU={kernelName:di,backendName:"webgl",kernelFunc:oU};const iU=`
  float s = sign(a) * sign(b);
  int ia = round(a);
  int ib = round(b);
  if (ib != 0) {
    // Windows (D3D) wants guaranteed non-zero int division at compile-time.
    return float(idiv(ia, ib, s));
  } else {
    return NAN;
  }
`,aU=`
  ivec4 ia = round(a);
  ivec4 ib = round(b);
  bvec4 cond = notEqual(ib, ivec4(0));
  ivec4 result = ivec4(0);
  vec4 s = sign(a) * sign(b);

  // Windows (D3D) wants guaranteed non-zero int division at compile-time.
  if (cond[0]) {
    result[0] = idiv(ia[0], ib[0], s[0]);
  }
  if (cond[1]) {
    result[1] = idiv(ia[1], ib[1], s[1]);
  }
  if (cond[2]) {
    result[2] = idiv(ia[2], ib[2], s[2]);
  }
  if (cond[3]) {
    result[3] = idiv(ia[3], ib[3], s[3]);
  }
  return vec4(result);
`,lU=ve({opSnippet:iU,packedOpSnippet:aU,dtype:"int32"}),cU={kernelName:pi,backendName:"webgl",kernelFunc:lU};class uU{constructor(t){this.variableNames=["A"];const e=Le(),[s,o]=t;this.outputShape=t,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${o}.0, ${s}.0);

        vec4 values = ${e.texture2D}(A, uv);
        float value;
        if (depth == 0) {
          value = values.r;
        } else if (depth == 1) {
          value = values.g;
        } else if (depth == 2) {
          value = values.b;
        } else if (depth == 3) {
          value = values.a;
        }

        setOutput(floor(value * 255.0 + 0.5));
      }
    `}}class hU{constructor(t){this.variableNames=["A"],this.packedInputs=!1,this.packedOutput=!0;const e=Le(),[s,o]=t;this.outputShape=t,this.userCode=`
      void main() {
        ivec3 coords = getOutputCoords();
        int texR = coords[0];
        int texC = coords[1];
        int depth = coords[2];

        vec4 result = vec4(0.);

        for(int row=0; row<=1; row++) {
          for(int col=0; col<=1; col++) {
            texC = coords[1] + row;
            depth = coords[2] + col;

            vec2 uv = (vec2(texC, texR) + halfCR) /
                       vec2(${o}.0, ${s}.0);
            vec4 values = ${e.texture2D}(A, uv);
            float value;
            if (depth == 0) {
              value = values.r;
            } else if (depth == 1) {
              value = values.g;
            } else if (depth == 2) {
              value = values.b;
            } else if (depth == 3) {
              value = values.a;
            }

            result[row * 2 + col] = floor(value * 255.0 + 0.5);
          }
        }

        ${e.output} = result;
      }
    `}}const dU={kernelName:D$,backendName:"webgl",kernelFunc:pU};let Lo,du=B().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");function pU(n){const{inputs:t,backend:e,attrs:s}=n;let{pixels:o}=t;const{numChannels:r}=s,i=typeof HTMLVideoElement<"u"&&o instanceof HTMLVideoElement,a=typeof HTMLImageElement<"u"&&o instanceof HTMLImageElement,[l,c]=i?[o.videoWidth,o.videoHeight]:[o.width,o.height],u=[c,l],h=[c,l,r];if(a||i){const m=B().getBool("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU");(Lo==null||m!==du)&&(du=m,Lo=document.createElement("canvas").getContext("2d",{willReadFrequently:du})),Lo.canvas.width=l,Lo.canvas.height=c,Lo.drawImage(o,0,0,l,c),o=Lo.canvas}const d=e.makeTensorInfo(u,"int32");e.texData.get(d.dataId).usage=en.PIXELS,e.gpgpu.uploadPixelDataToTexture(e.getTexture(d.dataId),o);const p=B().getBool("WEBGL_PACK")?new hU(h):new uU(h),f=e.runWebGLProgram(p,[d],"int32");return e.disposeData(d.dataId),f}function fU(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dataFormat:u,dilations:h,dimRoundingMode:d,activation:p,leakyreluAlpha:f}=s,m=us(u),g=Ce(o.shape,r.shape,l,h,c,d,!1,m);let x;const b=[],w=i!=null,y=a!=null,$=p==="leakyrelu",v=()=>{const T=[o,r],k=(S,C)=>{if(C==="NCHW"&&S.shape.length===1&&S.shape[0]!==1){const D=et({inputs:{x:S},backend:e,attrs:{shape:[S.shape[0],1,1]}});return b.push(D),D}return S};if(w&&T.push(k(i,u)),y&&T.push(k(a,u)),$){const S=e.makeTensorInfo([],"float32",as(f,"float32"));T.push(S),b.push(S)}return T};if(g.filterHeight===1&&g.filterWidth===1&&g.dilationHeight===1&&g.dilationWidth===1&&g.strideHeight===1&&g.strideWidth===1&&(g.padInfo.type==="SAME"||g.padInfo.type==="VALID"))x=qw({x:o,filter:r,convInfo:g,backend:e,bias:i,activation:p,preluActivationWeights:a,leakyreluAlpha:f});else if(g.strideWidth<=2&&m==="channelsLast"&&B().getBool("WEBGL_EXP_CONV")){const T=p?Hr(p,!0):null,k=new Hw(g,w,T,y,$),S=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],C=v();x=e.runWebGLProgram(k,C,"float32",S)}else if(B().getBool("WEBGL_CONV_IM2COL"))x=jw({x:o,filter:r,convInfo:g,backend:e,bias:i,activation:p,preluActivationWeights:a,leakyreluAlpha:f});else{const T=p?Hr(p,!1):null,k=new Gw(g,w,T,y,$),S=v();x=e.runWebGLProgram(k,S,"float32")}const N=et({inputs:{x},backend:e,attrs:{shape:g.outShape}});return b.push(x),b.forEach(T=>e.disposeIntermediateTensorInfo(T)),N}const mU={kernelName:Ua,backendName:"webgl",kernelFunc:fU};function gU(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,filter:r,bias:i,preluActivationWeights:a}=t,{strides:l,pad:c,dilations:u,dimRoundingMode:h,activation:d,leakyreluAlpha:p}=s,f=[];let m=u;m==null&&(m=[1,1]),I(Te(l,m),()=>`Error in depthwiseConv2d: Either strides or dilations must be 1. Got strides ${l} and dilations '${m}'`);const g=Ce(o.shape,r.shape,l,m,c,h,!0),x=B().getBool("WEBGL_PACK_DEPTHWISECONV")&&g.strideWidth<=2&&g.outChannels/g.inChannels===1,b=d?Hr(d,x):null,w=[o,r],y=i!=null,$=a!=null,v=d==="leakyrelu";if(y&&w.push(i),$&&w.push(a),v){const S=e.makeTensorInfo([],"float32",as(p,"float32"));w.push(S),f.push(S)}let N;x?N=new Yw(g,y,b,$,v):N=new Xw(g,y,b,$,v);const T=[[g.padInfo.top,g.padInfo.left],[g.strideHeight,g.strideWidth],[g.dilationHeight,g.dilationWidth],[g.inHeight,g.inWidth]],k=e.runWebGLProgram(N,w,"float32",T);return f.forEach(S=>e.disposeIntermediateTensorInfo(S)),k}const xU={kernelName:dg,backendName:"webgl",kernelFunc:gU};class bU{constructor(t,e,s,o){this.sliceDim=t,this.strides=e,this.paramsShape=o,this.variableNames=["x","indices"],this.outputShape=s;const r=Ot(s.length);let i=`
    int index;`;for(let a=0;a<this.sliceDim;a++)i+=`
          index = round(getIndices(coords[0], ${a}));
          out_of_bounds = out_of_bounds || index < 0;
          out_of_bounds = out_of_bounds || index >= ${this.paramsShape[a]};
          flattenIndex += index * ${this.strides[a]};`;this.userCode=`
         void main() {
          ${r} coords = getOutputCoords();
          int flattenIndex = 0;
          bool out_of_bounds = false;

          ${i}

          setOutput(out_of_bounds ? 0.0 : getX(flattenIndex, coords[1]));
        }
      `}}function yU(n){const{inputs:t,backend:e}=n,{params:s,indices:o}=t,r=o.shape,i=r[r.length-1],a=U(s.shape),[l,c,u,h]=Sc(s,o),d=et({inputs:{x:o},backend:e,attrs:{shape:[c,i]}}),p=et({inputs:{x:s},backend:e,attrs:{shape:[U(s.shape)/u,u]}});if(e.shouldExecuteOnCPU([s,o])||s.dtype==="string"){const x=e.readSync(o.dataId),b=e.bufferSync(s),w=Cz(x,b,s.dtype,c,i,u,h,s.shape,a);return e.makeTensorInfo(l,s.dtype,w.values)}const f=new bU(i,h,[c,u],s.shape),m=e.runWebGLProgram(f,[p,d],p.dtype),g=et({inputs:{x:m},backend:e,attrs:{shape:l}});return e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(m),g}const wU={kernelName:kh,backendName:"webgl",kernelFunc:yU};class $U{constructor(t,e){this.variableNames=["A","indices"],this.outputShape=e,this.rank=e.length;const s=Ot(this.rank),o=CU(t);this.userCode=`
      void main() {
        ${s} resRC = getOutputCoords();
        int index = int(getIndices(resRC.x, resRC.z));
        float inBounds = (index >= 0) && (index < ${t[2]}) ? 1.0 : 0.0;
        setOutput(inBounds * getA(${o}));
      }
    `}}function CU(n,t){const e=["resRC.x","resRC.y","resRC.z","resRC.w"],s=[];for(let o=0;o<n.length;o++)o===2?s.push("index"):s.push(`${e[o]}`);return s.join()}function Qw(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,indices:r}=t,{axis:i,batchDims:a}=s,l=$t(i,o.shape)[0];if(B().get("DEBUG")){const b=e.readSync(r.dataId),w=o.shape[l];for(let y=0;y<b.length;++y){const $=b[y];I($<=w-1&&$>=0,()=>`GatherV2: the index value ${$} is not in [0, ${w-1}]`)}}const c=up(o,r,l,a),u=U(r.shape),h=[],d=et({inputs:{x:o},backend:e,attrs:{shape:[c.batchSize,c.outerSize,c.dimSize,c.sliceSize]}}),p=et({inputs:{x:r},backend:e,attrs:{shape:[c.batchSize,u/c.batchSize]}});h.push(d),h.push(p);const f=[c.batchSize,c.outerSize,u/c.batchSize,c.sliceSize];if(e.shouldExecuteOnCPU([o,r])||o.dtype==="string"){const b=e.bufferSync(p),w=e.bufferSync(d),y=Iz(w,b,f);return h.forEach($=>e.disposeIntermediateTensorInfo($)),e.makeTensorInfo(c.outputShape,y.dtype,y.values)}const m=new $U(d.shape,f),g=e.runWebGLProgram(m,[d,p],d.dtype);h.push(g);const x=et({inputs:{x:g},backend:e,attrs:{shape:c.outputShape}});return h.forEach(b=>e.disposeIntermediateTensorInfo(b)),x}const IU={kernelName:_l,backendName:"webgl",kernelFunc:Qw};const vU="return float(a > b);",kU=`
  return vec4(greaterThan(a, b));
`,SU=ve({opSnippet:vU,packedOpSnippet:kU,cpuKernelImpl:vz,dtype:"bool"}),NU={kernelName:Ol,backendName:"webgl",kernelFunc:SU};const TU="return float(a >= b);",EU=`
  return vec4(greaterThanEqual(a, b));
`,RU=ve({opSnippet:TU,packedOpSnippet:EU,dtype:"bool",cpuKernelImpl:kz}),AU={kernelName:fi,backendName:"webgl",kernelFunc:RU};function DU(n){const{inputs:t,backend:e}=n,{input:s}=t;return Jw(s,!0,e)}const FU={kernelName:Sh,backendName:"webgl",kernelFunc:DU};const _U="return float(!isnan(x) && !isinf(x));",OU=St({opSnippet:_U,dtype:"bool"}),LU={kernelName:gi,backendName:"webgl",kernelFunc:OU};const MU="return float(isinf(x));",PU=St({opSnippet:MU,dtype:"bool"}),zU={kernelName:xi,backendName:"webgl",kernelFunc:PU};const BU="return float(isnan(x));",VU=St({opSnippet:BU,dtype:"bool"}),WU={kernelName:bi,backendName:"webgl",kernelFunc:VU};const UU="return float(a < b);",GU=`
  return vec4(lessThan(a, b));
`,HU=ve({opSnippet:UU,packedOpSnippet:GU,cpuKernelImpl:Sz,dtype:"bool"}),qU={kernelName:Ml,backendName:"webgl",kernelFunc:HU};const jU="return float(a <= b);",KU=`
  return vec4(lessThanEqual(a, b));
`,XU=ve({opSnippet:jU,packedOpSnippet:KU,cpuKernelImpl:Nz,dtype:"bool"}),YU={kernelName:Pl,backendName:"webgl",kernelFunc:XU};function ZU(n){const{backend:t,attrs:e}=n,{start:s,stop:o,num:r}=e,i=Tz(s,o,r);return t.makeTensorInfo([i.length],"float32",i)}const JU={kernelName:ng,backendName:"webgl",kernelFunc:ZU};const QU=pr+`
  return x < 0.0 ? 0./0. : log(x);
`,tG=`
  vec4 result = log(x);
  bvec4 isNaN = isnan(x);
  result.r = isNaN.r ? x.r : (x.r < 0.0 ? 0./0. : result.r);
  result.g = isNaN.g ? x.g : (x.g < 0.0 ? 0./0. : result.g);
  result.b = isNaN.b ? x.b : (x.b < 0.0 ? 0./0. : result.b);
  result.a = isNaN.a ? x.a : (x.a < 0.0 ? 0./0. : result.a);
  return result;
`,eG=St({opSnippet:QU,packedOpSnippet:tG,cpuKernelImpl:Ez}),nG={kernelName:yi,backendName:"webgl",kernelFunc:eG};const sG=pr+`
  return log(1.0 + x);
`,oG=St({opSnippet:sG}),rG={kernelName:wi,backendName:"webgl",kernelFunc:oG};const iG="return float(a >= 1.0 && b >= 1.0);",aG=`
  return vec4(
    vec4(greaterThanEqual(a, vec4(1.0))) *
    vec4(greaterThanEqual(b, vec4(1.0))));
`,lG=ve({opSnippet:iG,packedOpSnippet:aG,dtype:"bool"}),cG={kernelName:zl,backendName:"webgl",kernelFunc:lG};const uG="return float(!(x >= 1.0));",hG=St({opSnippet:uG}),dG={kernelName:Bl,backendName:"webgl",kernelFunc:hG};const pG="return float(a >= 1.0 || b >= 1.0);",fG=`
  return min(
    vec4(greaterThanEqual(a, vec4(1.0))) +
    vec4(greaterThanEqual(b, vec4(1.0))),
    vec4(1.0));
`,mG=ve({opSnippet:pG,packedOpSnippet:fG,dtype:"bool"}),gG={kernelName:Vl,backendName:"webgl",kernelFunc:mG};class xG{constructor(t,e,s,o,r){this.variableNames=["x"],this.outputShape=[];const i=e,a=t[3]-1;this.outputShape=t;let l;const c=`float(${s}) + float(${o}) * sum`;r===.5?l=`inversesqrt(${c})`:r===1?l=`1.0/(${c})`:l=`exp(log(${c}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];
        int d = coords[3];
        float x = getX(b, r, c, d);
        float sum = 0.0;
        for (int j = -${i}; j <= ${i}; j++) {
          int idx = d + j;
          if (idx >= 0 && idx <=  ${a}) {
            float z = getX(b, r, c, idx);
            sum += z * z;
          }
        }
        float val = x * ${l};
        setOutput(val);
      }
    `}}class bG{constructor(t,e,s,o,r){this.variableNames=["x"],this.outputShape=[],this.packedInputs=!0,this.packedOutput=!0;const i=e,a=t[3]-1;this.outputShape=t;let l;const c=`float(${s}) + float(${o}) * sum`;r===.5?l=`inversesqrt(${c})`:r===1?l=`1.0/(${c})`:l=`exp(log(${c}) * float(-${r}));`,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords.x;
        int r = coords.y;
        int c = coords.z;
        int d = coords.w;

        bool hasNextCol = d < ${this.outputShape[3]};
        bool hasNextRow = c < ${this.outputShape[2]};

        vec4 sum = vec4(0.);
        vec4 xFragAtOutputCoords = getX(b, r, c, d);

        vec4 xAtOutputCoords = vec4(
          getChannel(xFragAtOutputCoords, vec2(c, d)),
          hasNextCol ?
            getChannel(xFragAtOutputCoords, vec2(c, d + 1)) : 0.0,
          hasNextRow ?
            getChannel(xFragAtOutputCoords , vec2(c + 1, d)) : 0.0,
          (hasNextRow && hasNextCol) ?
            getChannel(xFragAtOutputCoords, vec2(c + 1, d + 1)) : 0.0
        );

        int firstChannel = d - ${i};
        vec2 cache = vec2(0.);
        if(firstChannel >= 0){
          vec4 firstChannelFrag = getX(b, r, c, firstChannel);
          cache.x = getChannel(firstChannelFrag, vec2(c, firstChannel));
            if(hasNextRow){
              cache.y = getChannel(firstChannelFrag, vec2(c + 1, firstChannel));
            }
        }

        ivec2 depth = ivec2(d, d + 1);
        for (int j = - ${i}; j <= ${i}; j++) {
          ivec2 idx = depth + j;
          bvec2 aboveLowerBound = greaterThanEqual(idx, ivec2(0));
          bvec2 belowUpperBound = lessThanEqual(idx, ivec2(${a}));

          bool depthInRange = aboveLowerBound.x && belowUpperBound.x;
          bool depthPlusOneInRange = aboveLowerBound.y && belowUpperBound.y;

          if(depthInRange || depthPlusOneInRange){
            vec4 z = vec4(0.);
            vec4 xFragAtCurrentDepth;
            z.xz = cache.xy;
            if(depthPlusOneInRange && hasNextCol){
              xFragAtCurrentDepth = idx.y != d ?
                getX(b, r, c, idx.y) : xFragAtOutputCoords;
              z.y = getChannel(xFragAtCurrentDepth, vec2(c, idx.y));
              if(hasNextRow){
                z.w = getChannel(xFragAtCurrentDepth, vec2(c + 1, idx.y));
              }
            }
            cache.xy = z.yw;
            sum += z * z;
          }
        }
        vec4 result = xAtOutputCoords * ${l};
        setOutput(result);
      }
    `}}const yG=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{depthRadius:r,bias:i,alpha:a,beta:l}=s,c=B().getBool("WEBGL_PACK_NORMALIZATION")?new bG(o.shape,r,i,a,l):new xG(o.shape,r,i,a,l);return e.runWebGLProgram(c,[o],o.dtype)},wG={kernelName:Wl,backendName:"webgl",kernelFunc:yG};class $G{constructor(t,e,s,o,r){this.variableNames=["inputImage","outputImage","dy"],this.outputShape=[],this.outputShape=t,this.depth=t[3],this.depthRadius=e,this.bias=s,this.alpha=o,this.beta=r,this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int r = coords[1];
        int c = coords[2];

        float result = 0.0;
        for (int d = 0; d < ${this.depth}; ++d) {
          int depthBegin = int(max(0.0, float(d - ${e})));
          int depthEnd = int(min(float(${this.depth}),
              float(d + ${e} + 1)));

          const int MIN_DEPTH_BEGIN = 0;
          const int MAX_DEPTH_END = ${this.depth};

          float norm = 0.0;
          for (int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k) {
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd) {
              norm += getInputImage(b, r, c, k) * getInputImage(b, r, c, k);
            }
            else {
              break;
            }
          }

          norm = float(${o}) * norm + float(${s});

          for(int k = MIN_DEPTH_BEGIN; k < MAX_DEPTH_END; ++k){
            if (k < depthBegin){
              continue;
            }
            else if (k >= depthBegin && k < depthEnd){
              float dyi = -2.0 * float(${o})
                * float(${r})
                * getInputImage(b, r, c, k) * getOutputImage(b, r, c, d)
                / norm;
              if (k == d) {
                dyi += pow(norm, -1.0 * ${r});
              }
              if (k == coords[3]) {
                dyi *= getDy(b, r, c, d);
                result += dyi;
              }
            }
            else {
              break;
            }
          }
      }
      setOutput(result);
      }
    `}}const CG=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o,y:r,dy:i}=t,{depthRadius:a,bias:l,alpha:c,beta:u}=s,h=new $G(o.shape,a,l,c,u);return e.runWebGLProgram(h,[o,r,i],o.dtype)},IG={kernelName:Th,backendName:"webgl",kernelFunc:CG};function vG(n,t,e,s){const o=U(t),i=U(n.shape)/o,a=et({inputs:{x:n},attrs:{shape:[i,o]},backend:s}),l=Fo(a,n.dtype,"max",s),c=et({inputs:{x:l},attrs:{shape:e},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(l),c}function t$(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reductionIndices:r,keepDims:i}=s,a=o.shape.length,l=$t(r,o.shape);let c=l;const u=Xt(c,a),h=u!=null,d=e.shouldExecuteOnCPU([o]);let p=o;if(h){if(d){const w=e.texData.get(p.dataId).values,y=new Array(a);for(let N=0;N<y.length;N++)y[N]=o.shape[u[N]];const $=tf(w,o.shape,o.dtype,u,y);p=e.makeTensorInfo(y,o.dtype);const v=e.texData.get(p.dataId);v.values=$}else p=jc(o,u,e);c=ne(c.length,a)}Ie("max",c,a);const[f,m]=me(p.shape,c);let g=f;i&&(g=ee(f,l));let x;if(d){const w=e.texData.get(p.dataId).values,y=Rz(w,U(m),g,o.dtype);x=e.makeTensorInfo(g,o.dtype);const $=e.texData.get(x.dataId);$.values=y}else x=vG(p,m,g,e);return h&&e.disposeIntermediateTensorInfo(p),x}const kG={kernelName:Ul,backendName:"webgl",kernelFunc:t$};const SG=ef+`
  return max(a, b);
`,NG=`
  vec4 result = vec4(max(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Do+`
  return result;
`,TG=ve({opSnippet:SG,packedOpSnippet:NG,cpuKernelImpl:Az}),EG={kernelName:$i,backendName:"webgl",kernelFunc:TG};function RG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;ar(o,"maxPool");const{filterSize:r,strides:i,pad:a,dimRoundingMode:l}=s,c=1;I(Te(i,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${i} and dilations '${c}'`);const u=xn(o.shape,r,i,c,a,l);if(u.filterWidth===1&&u.filterHeight===1&&Tt(u.inShape,u.outShape))return Ye({inputs:{x:o},backend:e});const h=new qr(u,"max",!1);return e.runWebGLProgram(h,[o],o.dtype)}const AG={kernelName:Gl,backendName:"webgl",kernelFunc:RG};function DG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{filterSize:r,strides:i,pad:a,dataFormat:l,dimRoundingMode:c}=s,u=[1,1,1],h=cs(o.shape,r,i,u,a,c,l),d=new sf(h,"max",!1);return e.runWebGLProgram(d,[o],o.dtype)}const FG={kernelName:Hl,backendName:"webgl",kernelFunc:DG};class _G{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideHeight,s=t.strideWidth,o=t.dilationHeight,r=t.effectiveFilterHeight,i=t.effectiveFilterWidth,a=r-1-t.padInfo.top,l=i-1-t.padInfo.left,c=r*i-1;this.userCode=`
      const ivec2 pads = ivec2(${a}, ${l});

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];

        ivec2 dyRCCorner = coords.yz - pads;
        int dyRCorner = dyRCCorner.x;
        int dyCCorner = dyRCCorner.y;

        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;
        for (int wR = 0; wR < ${r};
          wR += ${o}) {
          float dyR = float(dyRCorner + wR) / ${e}.0;

          if (dyR < 0.0 || dyR >= ${t.outHeight}.0 || fract(dyR) > 0.0) {
            continue;
          }
          int idyR = int(dyR);

          for (int wC = 0; wC < ${i}; wC++) {
            float dyC = float(dyCCorner + wC) / ${s}.0;

            if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                fract(dyC) > 0.0) {
              continue;
            }
            int idyC = int(dyC);

            float dyValue = getDy(b, idyR, idyC, d);
            int maxPosValue = ${c} - int(getMaxPos(b, idyR, idyC, d));

            // Get the current value, check it against the value from the
            // position matrix.
            int curPosValue = wR * ${i} + wC;
            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

            dotProd += dyValue * mask;
          }
        }
        setOutput(dotProd);
      }
    `}}class OG{constructor(t){this.variableNames=["dy","maxPos"],this.outputShape=t.inShape;const e=t.strideDepth,s=t.strideHeight,o=t.strideWidth,r=t.dilationDepth,i=t.dilationHeight,a=t.dilationWidth,l=t.effectiveFilterDepth,c=t.effectiveFilterHeight,u=t.effectiveFilterWidth,h=l-1-t.padInfo.front,d=c-1-t.padInfo.top,p=u-1-t.padInfo.left,f=l*c*u-1;this.userCode=`
      const ivec3 pads = ivec3(${h}, ${d}, ${p});

      void main() {
        ivec5 coords = getOutputCoords();
        int batch = coords.x;
        int ch = coords.u;

        ivec3 dyCorner = ivec3(coords.y, coords.z, coords.w) - pads;
        int dyDCorner = dyCorner.x;
        int dyRCorner = dyCorner.y;
        int dyCCorner = dyCorner.z;

        // Convolve dy(?, ?, ?, ch) with pos mask(:, :, :, d) to get
        // dx(xD, xR, xC, ch).
        // ? = to be determined. : = across all values in that axis.
        float dotProd = 0.0;

        for (int wD = 0; wD < ${l};
           wD += ${r}) {
          float dyD = float(dyDCorner + wD) / ${e}.0;

          if (dyD < 0.0 || dyD >= ${t.outDepth}.0 || fract(dyD) > 0.0) {
            continue;
          }
          int idyD = int(dyD);

          for (int wR = 0; wR < ${c};
              wR += ${i}) {
            float dyR = float(dyRCorner + wR) / ${s}.0;

            if (dyR < 0.0 || dyR >= ${t.outHeight}.0 ||
                fract(dyR) > 0.0) {
              continue;
            }
            int idyR = int(dyR);

            for (int wC = 0; wC < ${u};
                wC += ${a}) {
              float dyC = float(dyCCorner + wC) / ${o}.0;

              if (dyC < 0.0 || dyC >= ${t.outWidth}.0 ||
                  fract(dyC) > 0.0) {
                continue;
              }
              int idyC = int(dyC);

              float dyValue = getDy(batch, idyD, idyR, idyC, ch);
              int maxPosValue = ${f} -
                  int(getMaxPos(batch, idyD, idyR, idyC, ch));

              // Get the current value, check it against the value from the
              // position matrix.
              int curPosValue =
                  wD * ${c} * ${u} +
                  wR * ${u} + wC;
              float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);

              dotProd += dyValue * mask;
            }
          }
        }
        setOutput(dotProd);
      }
    `}}function LG(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r}=t,i=r,{filterSize:a,strides:l,pad:c,dimRoundingMode:u}=s,h=[1,1,1],d=cs(i.shape,a,l,h,c,u),p=new sf(d,"max",!0),f=e.runWebGLProgram(p,[i],i.dtype),m=new OG(d),g=e.runWebGLProgram(m,[o,f],i.dtype);return e.disposeIntermediateTensorInfo(f),g}const MG={kernelName:Rh,backendName:"webgl",kernelFunc:LG};function PG(n){const{inputs:t,backend:e,attrs:s}=n,{dy:o,input:r,output:i}=t,a=r;ar([r,i],"maxPoolGrad");const{filterSize:l,strides:c,pad:u,dimRoundingMode:h}=s,d=xn(a.shape,l,c,1,u,h),p=!0,f=new qr(d,"max",p),m=e.runWebGLProgram(f,[a],a.dtype),g=new _G(d),x=e.runWebGLProgram(g,[o,m],a.dtype);return e.disposeIntermediateTensorInfo(m),x}const zG={kernelName:Eh,backendName:"webgl",kernelFunc:PG};function BG(n,t,e,s){let o=new qr(e,"max",!1);const r=s.runWebGLProgram(o,[n],"float32");o=new qr(e,"max",!0,!0,t);const i=s.runWebGLProgram(o,[n],"float32");return[r,i]}const VG={kernelName:sg,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{filterSize:o,strides:r,pad:i,includeBatchInIndex:a}=t,l=e;I(s.shape.length===4,()=>`Error in maxPool: input must be rank 4 but got rank ${s.shape.length}.`);const c=[1,1];I(Te(r,c),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${r} and dilations '${c}'`);const u=xn(s.shape,o,r,c,i),[h,d]=BG(s,a,u,l);return[h,d]}};function WG(n,t,e,s){const o=U(t),i=U(n.shape)/o,a=et({inputs:{x:n},attrs:{shape:[i,o]},backend:s}),l=Fo(a,"float32","mean",s),c=et({inputs:{x:l},attrs:{shape:e},backend:s});return s.disposeIntermediateTensorInfo(a),s.disposeIntermediateTensorInfo(l),c}const UG={kernelName:ql,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{x:s}=n,{keepDims:o,axis:r}=t,i=e,a=s.shape.length,l=$t(r,s.shape);let c=l;const u=Xt(c,a),h=u!=null,d=i.shouldExecuteOnCPU([s]),p=[];let f=s;if(h){if(d){const y=i.texData.get(f.dataId).values,$=new Array(a);for(let T=0;T<$.length;T++)$[T]=s.shape[u[T]];const v=tf(y,s.shape,s.dtype,u,$);f=i.makeTensorInfo($,s.dtype);const N=i.texData.get(f.dataId);N.values=v}else f=jc(s,u,i);p.push(f),c=ne(c.length,a)}Ie("sum",c,a);const[m,g]=me(f.shape,c);let x=m;o&&(x=ee(m,l));const b=WG(f,g,x,i);for(const w of p)i.disposeIntermediateTensorInfo(w);return b}};function GG(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=$t(r,o.shape);let c=l;const u=Xt(c,a);let h=o;u!=null&&(h=_e({inputs:{x:o},backend:e,attrs:{perm:u}}),c=ne(c.length,o.shape.length)),Ie("min",c,a);const[d,p]=me(h.shape,c),f=U(p),m=et({inputs:{x:h},backend:e,attrs:{shape:[-1,f]}}),g=Fo(m,m.dtype,"min",e);let x;if(i){const b=ee(d,l);x=et({inputs:{x:g},backend:e,attrs:{shape:b}})}else x=et({inputs:{x:g},backend:e,attrs:{shape:d}});return e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(g),u!=null&&e.disposeIntermediateTensorInfo(h),x}const HG={kernelName:jl,backendName:"webgl",kernelFunc:GG};const qG=ef+`
  return min(a, b);
`,jG=`
  vec4 result = vec4(min(a, b));
  bvec4 isNaNA = isnan(a);
  bvec4 isNaNB = isnan(b);
  bvec4 isNaN = bvec4(isNaNA.x || isNaNB.x, isNaNA.y || isNaNB.y, isNaNA.z || isNaNB.z, isNaNA.w || isNaNB.w);
  `+Do+`
  return result;
`,KG=ve({opSnippet:qG,packedOpSnippet:jG,cpuKernelImpl:Dz}),XG={kernelName:Ci,backendName:"webgl",kernelFunc:KG};class YG{constructor(t,e,s){this.variableNames=["x"],this.outputShape=e.map((u,h)=>u[0]+t[h]+u[1]);const o=t.length,r=Ot(o),i=e.map(u=>u[0]).join(","),a=e.map((u,h)=>u[0]+t[h]).join(","),l=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o),c=s==="reflect"?0:1;if(o===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start) {
            outC = start * 2 - outC - ${c};
          } else if(outC >= end) {
            outC = (end - 1) * 2 - outC + ${c};
          }
          setOutput(getX(outC - start));
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${i});
      ${r} end = ${r}(${a});

      void main() {
        ${r} outC = getOutputCoords();
        for (int i = 0; i < ${o}; i++) {
          if (outC[i] < start[i]) {
            outC[i] = start[i] * 2 - outC[i] - ${c};
          } else if(outC[i] >= end[i]) {
            outC[i] = (end[i] - 1) * 2 - outC[i] + ${c};
          }
        }
        ${r} coords = outC - start;
        setOutput(getX(${l}));
      }
    `}}class ZG{constructor(t,e,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=e.map((f,m)=>f[0]+t[m]+f[1]);const o=t.length,r=Ot(o),i=e.map(f=>f[0]).join(","),a=e.map((f,m)=>f[0]+t[m]).join(","),l=Fe("rc",o),c=Fe("source",o),u=`${l[o-1]} < ${this.outputShape[o-1]}`,h=o===1?"source":`vec2(${c.slice(-2).join()})`,d=s==="reflect"?0:1;let p="";if(o===1){const f=`
        ${r} source = rc;
        if (source < start) {
          source = start * 2 - source - ${d};
        } else if (source >= end) {
          source = (end - 1) * 2 - source + ${d};
        }
        source -= start;
      `;p=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${c.join()}), ${h});
        ${l[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${c.join()}), ${h});
        }
      `}else{const f=`
        ${r} source = rc;
        ${r} lt = ${r}(lessThan(source, start));
        ${r} gte = ${r}(greaterThanEqual(source, end));
        ${r} orig = 1 - (lt + gte);
        source = orig * source +
                lt * (start * 2 - source - ${d}) +
                gte * ((end - 1) * 2 - source + ${d});
        source -= start;
      `;p=`
        ${r} rc = outputLoc;
        ${f}
        result[0] = getChannel(getX(${c.join()}), ${h});
        ${l[o-1]} += 1;
        if(${u}) {
          ${f}
          result[1] = getChannel(getX(${c.join()}), ${h});
        }
        rc = outputLoc;
        ${l[o-2]} += 1;
        if(${l[o-2]} < ${this.outputShape[o-2]}) {
          ${f}
          result[2] = getChannel(getX(${c.join()}), ${h});
          ${l[o-1]} += 1;
          if(${u}) {
            ${f}
            result[3] = getChannel(getX(${c.join()}), ${h});
          }
        }
      `}this.userCode=`
      const ${r} start = ${r}(${i});
      const ${r} end = ${r}(${a});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${p}
        setOutput(result);
      }
    `}}const JG=({inputs:n,backend:t,attrs:e})=>{const{x:s}=n,{paddings:o,mode:r}=e,i=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new ZG(s.shape,o,r):new YG(s.shape,o,r);return t.runWebGLProgram(i,[s],s.dtype)},QG={kernelName:Kl,backendName:"webgl",kernelFunc:JG};const tH=`if (b == 0.0) return NAN;
  return mod(a, b);`,eH=`
  vec4 result = mod(a, b);
  bvec4 isNaN = equal(b, vec4(0.0));
  `+Do+`
  return result;
`,nH=ve({opSnippet:tH,packedOpSnippet:eH}),sH={kernelName:Ii,backendName:"webgl",kernelFunc:nH};class oH{constructor(t,e,s){this.variableNames=["probs"],this.customUniforms=[{name:"seed",type:"float"}],this.outputShape=[t,s],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];

        float r = random(seed);
        float cdf = 0.0;

        for (int i = 0; i < ${e-1}; i++) {
          cdf += getProbs(batch, i);

          if (r < cdf) {
            setOutput(float(i));
            return;
          }
        }

        // If no other event happened, last event happened.
        setOutput(float(${e-1}));
      }
    `}}const rH=`
if (a == b) {
  return 1.0;
};
return a / b;`,iH=`
  // vec4 one = vec4(equal(a, b));
  // return one + (vec4(1.0) - one) * a / b;
  vec4 result = a / b;
  if(a.x == b.x) {
    result.x = 1.;
  }
  if(a.y == b.y) {
    result.y = 1.;
  }
  if(a.z == b.z) {
    result.z = 1.;
  }
  if(a.w == b.w) {
    result.w = 1.;
  }

  return result;
`,e$=ve({opSnippet:rH,packedOpSnippet:iH,checkOutOfBounds:!0}),aH={kernelName:ai,backendName:"webgl",kernelFunc:e$};const Bm="return a - b;",n$=ve({opSnippet:Bm,packedOpSnippet:Bm,supportsComplex:!0,cpuKernelImpl:Qz}),lH={kernelName:zi,backendName:"webgl",kernelFunc:n$};function s$(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{dim:r}=s,i=$t([r],o.shape),a=t$({inputs:{x:o},backend:e,attrs:{reductionIndices:i,keepDims:!1}}),l=ee(a.shape,i),c=et({inputs:{x:a},backend:e,attrs:{shape:l}}),u=n$({inputs:{a:o,b:c},backend:e}),h=Zw({inputs:{x:u},backend:e}),d=Kc({inputs:{x:h},backend:e,attrs:{axis:i,keepDims:!1}}),p=et({inputs:{x:d},backend:e,attrs:{shape:l}}),f=e$({inputs:{a:h,b:p},backend:e});return e.disposeIntermediateTensorInfo(a),e.disposeIntermediateTensorInfo(c),e.disposeIntermediateTensorInfo(u),e.disposeIntermediateTensorInfo(h),e.disposeIntermediateTensorInfo(d),e.disposeIntermediateTensorInfo(p),f}const cH={kernelName:dc,backendName:"webgl",kernelFunc:s$};function uH(n){const{inputs:t,backend:e,attrs:s}=n,{logits:o}=t,{numSamples:r,seed:i,normalized:a}=s,l=a?o:s$({inputs:{logits:o},backend:e,attrs:{dim:o.shape.length-1}}),c=l.shape[0],u=l.shape[1],h=new oH(c,u,r),d=[[i]],p=e.runWebGLProgram(h,[l],"int32",d);return a||e.disposeIntermediateTensorInfo(l),p}const hH={kernelName:og,backendName:"webgl",kernelFunc:uH};const dH=yn+`
  return -x;
`,pH=`
  vec4 result = -x;
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`;function fH(n){const{inputs:t,backend:e}=n,{x:s}=t;if(e.shouldExecuteOnCPU([s])){const r=e.texData.get(s.dataId),[i,a]=_z(r.values,s.shape,s.dtype);return e.makeTensorInfo(a,s.dtype,i)}let o;return B().getBool("WEBGL_PACK_UNARY_OPERATIONS")?o=new Is(s.shape,pH):o=new zn(s.shape,dH),e.runWebGLProgram(o,[s],s.dtype)}const mH={kernelName:Xl,backendName:"webgl",kernelFunc:fH};const gH=Od;function xH(n){Qe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l}=s,c=e.readSync(o.dataId),u=e.readSync(r.dataId),{selectedIndices:h}=gH(c,u,i,a,l);return e.makeTensorInfo([h.length],"int32",new Int32Array(h))}const bH={kernelName:Ah,backendName:"webgl",kernelFunc:xH};const yH=Ld;function wH(n){Qe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,padToMaxOutputSize:c}=s,u=e.readSync(o.dataId),h=e.readSync(r.dataId),{selectedIndices:d,validOutputs:p}=yH(u,h,i,a,l,c);return[e.makeTensorInfo([d.length],"int32",new Int32Array(d)),e.makeTensorInfo([],"int32",new Int32Array([p]))]}const $H={kernelName:Dh,backendName:"webgl",kernelFunc:wH};const CH=Md;function IH(n){Qe("tf.nonMaxSuppression() in webgl locks the UI thread. Call tf.nonMaxSuppressionAsync() instead");const{inputs:t,backend:e,attrs:s}=n,{boxes:o,scores:r}=t,{maxOutputSize:i,iouThreshold:a,scoreThreshold:l,softNmsSigma:c}=s,u=e.readSync(o.dataId),h=e.readSync(r.dataId),d=i,p=a,f=l,m=c,{selectedIndices:g,selectedScores:x}=CH(u,h,d,p,f,m);return[e.makeTensorInfo([g.length],"int32",new Int32Array(g)),e.makeTensorInfo([x.length],"float32",new Float32Array(x))]}const vH={kernelName:Fh,backendName:"webgl",kernelFunc:IH};class kH{constructor(t,e,s,o){this.variableNames=["indices"],this.outputShape=[t,e],this.userCode=`
      void main() {
        ivec2 coords = getOutputCoords();
        int index = round(getIndices(coords.x));
        setOutput(mix(float(${o}), float(${s}),
                      float(index == coords.y)));
      }
    `}}const SH=n=>{const{inputs:t,backend:e,attrs:s}=n,{indices:o}=t,{dtype:r,depth:i,onValue:a,offValue:l}=s,c=U(o.shape),u=new kH(c,i,a,l),h=et({inputs:{x:o},backend:e,attrs:{shape:[c]}}),d=e.runWebGLProgram(u,[h],r);e.disposeIntermediateTensorInfo(h);const p=[...o.shape,i],f=et({inputs:{x:d},backend:e,attrs:{shape:p}});return e.disposeIntermediateTensorInfo(d),f},NH={kernelName:Jl,backendName:"webgl",kernelFunc:SH};function pl(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="complex64"){const o=ha({inputs:{input:s},backend:e}),r=pl({inputs:{x:o},backend:e}),i=Xc({inputs:{input:s},backend:e}),a=pl({inputs:{x:i},backend:e}),l=Gs({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return da({attrs:{shape:s.shape,dtype:s.dtype,value:s.dtype==="string"?"":0},backend:e})}const TH={kernelName:gc,backendName:"webgl",kernelFunc:pl};function o$(n){const{inputs:t,backend:e}=n,{x:s}=t;if(s.dtype==="string")throw new Error("onesLike is not supported under string dtype");if(s.dtype==="complex64"){const o=ha({inputs:{input:s},backend:e}),r=o$({inputs:{x:o},backend:e}),i=Xc({inputs:{input:s},backend:e}),a=pl({inputs:{x:i},backend:e}),l=Gs({inputs:{real:r,imag:a},backend:e});return e.disposeIntermediateTensorInfo(o),e.disposeIntermediateTensorInfo(r),e.disposeIntermediateTensorInfo(i),e.disposeIntermediateTensorInfo(a),l}else return da({attrs:{shape:s.shape,dtype:s.dtype,value:1},backend:e})}const EH={kernelName:Zl,backendName:"webgl",kernelFunc:o$};function RH(n){const{inputs:t,backend:e,attrs:s}=n,{axis:o}=s;if(t.length===1)return Xu({inputs:{input:t[0]},backend:e,attrs:{dim:o}});const r=t[0].shape,i=t[0].dtype;t.forEach(u=>{Oe(r,u.shape,"All tensors passed to stack must have matching shapes"),I(i===u.dtype,()=>"All tensors passed to stack must have matching dtypes")});const a=[],l=t.map(u=>{const h=Xu({inputs:{input:u},backend:e,attrs:{dim:o}});return a.push(h),h}),c=Uw({inputs:l,backend:e,attrs:{axis:o}});return a.forEach(u=>e.disposeIntermediateTensorInfo(u)),c}const AH={kernelName:Ql,backendName:"webgl",kernelFunc:RH};class DH{constructor(t,e,s){this.variableNames=["x"],this.customUniforms=[{name:"value",type:"float"}],this.outputShape=e.map((c,u)=>c[0]+t[u]+c[1]);const o=t.length,r=Ot(o),i=e.map(c=>c[0]).join(","),a=e.map((c,u)=>c[0]+t[u]).join(","),l=["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,o);if(o===1){this.userCode=`
        int start = ${i};
        int end = ${a};

        void main() {
          int outC = getOutputCoords();
          if (outC < start || outC >= end) {
            setOutput(value);
          } else {
            setOutput(getX(outC - start));
          }
        }
      `;return}this.userCode=`
      ${r} start = ${r}(${i});
      ${r} end = ${r}(${a});

      void main() {
        ${r} outC = getOutputCoords();
        if (any(lessThan(outC, start)) || any(greaterThanEqual(outC, end))) {
          setOutput(value);
        } else {
          ${r} coords = outC - start;
          setOutput(getX(${l}));
        }
      }
    `}}class FH{constructor(t,e,s){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0,this.customUniforms=[{name:"value",type:"float"}],this.outputShape=e.map((m,g)=>m[0]+t[g]+m[1]);const o=t.length,r=Ot(o),i=e.map(m=>m[0]).join(","),a=e.map((m,g)=>m[0]+t[g]).join(","),l=Fe("rc",o),c=Fe("source",o),u=`${l[o-1]} < ${this.outputShape[o-1]}`,h=o===1?"source":`vec2(${c.slice(-2).join()})`,d=[`${r} rc = outputLoc;`,`${l[o-1]} += 1;
       if(${u}) {
      `,o===1?"":`}
       rc = outputLoc;
       ${l[o-2]} += 1;
       if(${l[o-2]} < ${this.outputShape[o-2]}) {`,o===1?"":`  ${l[o-1]} += 1;
         if(${u}) {`],p=o===1?"rc < start || rc >= end":"any(lessThan(rc, start)) || any(greaterThanEqual(rc, end))";let f="";for(let m=0,g=o===1?2:4;m<g;m++)f+=`
        ${d[m]}
        if (${p}) {
          result[${m}] = float(value);
        } else {
          ${r} source = rc - start;
          result[${m}] = getChannel(getX(${c.join()}), ${h});
        }
      `;f+=o===1?"} ":"}}",this.userCode=`
      const ${r} start = ${r}(${i});
      const ${r} end = ${r}(${a});

      void main() {
        ${r} outputLoc = getOutputCoords();
        vec4 result = vec4(0.);
        ${f}
        setOutput(result);
      }
    `}}const r$=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{paddings:r,constantValue:i}=s;if(U(o.shape)===0){const c=r.map((u,h)=>u[0]+o.shape[h]+u[1]);return da({backend:e,attrs:{shape:c,value:i,dtype:o.dtype}})}const a=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new FH(o.shape,r,i):new DH(o.shape,r,i),l=[[i]];return e.runWebGLProgram(a,[o],o.dtype,l)},_H={kernelName:tc,backendName:"webgl",kernelFunc:r$};const OH=`
  if(a < 0.0 && floor(b) < b){
    return NAN;
  }
  if (b == 0.0) {
    return 1.0;
  }
  return (round(mod(b, 2.0)) != 1) ?
      pow(abs(a), b) : sign(a) * pow(abs(a), b);
`,LH=`
  // isModRound1 has 1 for components with round(mod(b, 2.0)) == 1, 0 otherwise.
  vec4 isModRound1 = vec4(equal(round(mod(b, 2.0)), ivec4(1)));
  vec4 multiplier = sign(a) * isModRound1 + (vec4(1.0) - isModRound1);
  vec4 result = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  bvec4 isExpZero = equal(b, vec4(0.0));
  result.r = isExpZero.r ? 1.0 : result.r;
  result.g = isExpZero.g ? 1.0 : result.g;
  result.b = isExpZero.b ? 1.0 : result.b;
  result.a = isExpZero.a ? 1.0 : result.a;

  bvec4 isNaN1 = lessThan(a, vec4(0.0));
  bvec4 isNaN2 = lessThan(floor(b), b);
  bvec4 isNaN = bvec4(isNaN1.x && isNaN2.x, isNaN1.y && isNaN2.y, isNaN1.z && isNaN2.z, isNaN1.w && isNaN2.w);
  `+Do+`
  return result;
`,MH=ve({opSnippet:OH,packedOpSnippet:LH}),PH={kernelName:ki,backendName:"webgl",kernelFunc:MH};function zH(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{axis:r,keepDims:i}=s,a=o.shape.length,l=[],c=$t(r,o.shape);let u=c;const h=Xt(u,a);let d=o;h!=null&&(d=_e({inputs:{x:o},backend:e,attrs:{perm:h}}),u=ne(u.length,a),l.push(d)),Ie("prod",u,a);let p;if(e.shouldExecuteOnCPU([d])){const f=e.texData.get(d.dataId).values,{outVals:m,outShape:g,outDtype:x}=Lz(d.shape,d.dtype,f,u);p=e.makeTensorInfo(g,x,m)}else{const[f,m]=me(d.shape,u),g=U(m),x=et({inputs:{x:d},backend:e,attrs:{shape:[-1,g]}}),b=Qh(o.dtype),w=Fo(x,b,"prod",e);p=et({inputs:{x:w},backend:e,attrs:{shape:f}}),l.push(x),l.push(w)}if(i){l.push(p);const f=ee(p.shape,c);p=et({inputs:{x:p},backend:e,attrs:{shape:f}})}return l.forEach(f=>e.disposeIntermediateTensorInfo(f)),p}const BH={kernelName:nc,backendName:"webgl",kernelFunc:zH};function VH(n){const{inputs:t,backend:e,attrs:s}=n,{paramsNestedSplits:o,paramsDenseValues:r,indices:i}=t,{outputRaggedRank:a}=s,l=o.map(x=>e.readSync(x.dataId)),c=o.map(x=>x.shape),u=e.readSync(r.dataId),h=e.readSync(i.dataId),[d,p,f]=Mz(l,c,u,r.shape,r.dtype,h,i.shape,a),m=d.map(x=>e.makeTensorInfo([x.length],"int32",x)),g=e.makeTensorInfo(f,r.dtype,p);return m.concat([g])}const WH={kernelName:rg,backendName:"webgl",kernelFunc:VH};function UH(n){const{inputs:t,backend:e}=n,{starts:s,limits:o,deltas:r}=t,i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=Pz(i,s.shape,s.dtype,a,o.shape,l,r.shape),h=e.makeTensorInfo([c.length],"int32",c),d=e.makeTensorInfo([u.length],s.dtype,u);return[h,d]}const GH={kernelName:ig,backendName:"webgl",kernelFunc:UH};function HH(n){const{inputs:t,backend:e,attrs:s}=n,{shape:o,values:r,defaultValue:i,rowPartitionTensors:a}=t,{rowPartitionTypes:l}=s,c=e.readSync(o.dataId),u=e.readSync(r.dataId),h=e.readSync(i.dataId),d=a.map(g=>e.readSync(g.dataId)),p=a.map(g=>g.shape),[f,m]=zz(c,o.shape,u,r.shape,r.dtype,h,i.shape,d,p,l);return e.makeTensorInfo(f,r.dtype,m)}const qH={kernelName:ag,backendName:"webgl",kernelFunc:HH};const i$=n=>{const{backend:t,attrs:e}=n,{start:s,stop:o,step:r,dtype:i}=e,a=Bz(s,o,r,i);return t.makeTensorInfo([a.length],i,a)},jH={kernelName:_h,backendName:"webgl",kernelFunc:i$};const KH="return 1.0 / x;",XH=St({opSnippet:KH}),YH={kernelName:Si,backendName:"webgl",kernelFunc:XH};const ZH=yn+`
  return (x < 0.0) ? 0.0 : x;
`,JH=`
  vec4 result = x * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,QH=St({opSnippet:ZH,packedOpSnippet:JH}),tq={kernelName:Ni,backendName:"webgl",kernelFunc:QH};const eq=yn+`
  return (x < 0.0) ? 0.0 : min(6.0, x);
`,nq=`
  vec4 result = min(x, vec4(6.)) * vec4(greaterThanEqual(x, vec4(0.0)));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,sq=St({opSnippet:eq,packedOpSnippet:nq}),oq={kernelName:Ti,backendName:"webgl",kernelFunc:sq};class rq{constructor(t,e,s,o,r){this.variableNames=["A"],this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s];let d;r?d="(vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC - vec2(0.5)":d="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${l}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec2 sourceFloorRC = ivec2(max(sourceFracIndexRC, vec2(0.0)));
        ivec2 sourceCeilRC = ivec2(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        float topLeft = getA(b, sourceFloorRC.x, sourceFloorRC.y, d);
        float bottomLeft = getA(b, sourceCeilRC.x, sourceFloorRC.y, d);
        float topRight = getA(b, sourceFloorRC.x, sourceCeilRC.y, d);
        float bottomRight = getA(b, sourceCeilRC.x, sourceCeilRC.y, d);

        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);

        float top = topLeft + (topRight - topLeft) * fracRC.y;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;
        float newValue = top + (bottom - top) * fracRC.x;

        setOutput(newValue);
      }
    `}}class iq{constructor(t,e,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s];let d;r?d="(vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC - vec3(0.5)":d="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${l}.0,
                                     ${l}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${d};

        // Compute the four integer indices.
        ivec3 sourceFloorRC = ivec3(max(sourceFracIndexRC, vec3(0.0)));
        ivec3 sourceCeilRC = ivec3(
          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${s-1};

        // In parallel, construct four corners for all four components in
        // packed 2x2 cell.
        vec4 topLeft = vec4(
          getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 bottomLeft = vec4(
          getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceFloorRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceFloorRC.z, d + 1) : 0.0);

        vec4 topRight = vec4(
          getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceFloorRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceFloorRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec4 bottomRight = vec4(
          getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d),
          hasNextCol ? getAValue(b, sourceCeilRC.x, sourceCeilRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceCeilRC.x, sourceCeilRC.z, d + 1) : 0.0);

        vec3 fracRC = sourceFracIndexRC - vec3(sourceFloorRC);

        vec4 top = mix(topLeft, topRight, fracRC.yyzz);
        vec4 bottom = mix(bottomLeft, bottomRight, fracRC.yyzz);
        vec4 newValue = mix(top, bottom, fracRC.x);

        setOutput(newValue);
      }
    `}}function aq(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s,[l,c]=a,u=B().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new iq(o.shape,l,c,r,i):new rq(o.shape,l,c,r,i);return e.runWebGLProgram(u,[o],"float32")}const lq={kernelName:rc,backendName:"webgl",kernelFunc:aq};class cq{constructor(t,e,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e;const[,o,r]=e,[,i,a]=t,l=[s&&i>1?o-1:o,s&&a>1?r-1:r],c=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=l[0]/c[0],h=l[1]/c[1],d=1/u,p=1/h,f=Math.ceil(d)*2+2,m=Math.ceil(p)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${f});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(startRLerp - float(winHeight / 2));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(startCLerp - float(winWidth / 2));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float dxR = float(dyR) * heightScale;
            int topDxRIndex = int(floor(dxR));
            int bottomDxRIndex = int(min(ceil(dxR), ${o-1}.0));
            float dxRLerp = dxR - float(topDxRIndex);
            float inverseDxRLerp = 1.0 - dxRLerp;

            float dxC = float(dyC) * widthScale;
            int leftDxCIndex = int(floor(dxC));
            int rightDxCIndex = int(min(ceil(dxC), ${r-1}.0));
            float dxCLerp = dxC - float(leftDxCIndex);
            float inverseDxCLerp = 1.0 - dxCLerp;

            if (r == topDxRIndex && c == leftDxCIndex) {
              // topLeft
              accumulator +=
                getDy(b, dyR, dyC, d) * inverseDxRLerp * inverseDxCLerp;
            }

            if (r == topDxRIndex && c == rightDxCIndex) {
              // topRight
              accumulator += getDy(b, dyR, dyC, d) * inverseDxRLerp * dxCLerp;
            }

            if (r == bottomDxRIndex && c == leftDxCIndex) {
              // bottomLeft
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * inverseDxCLerp;
            }

            if (r == bottomDxRIndex && c == rightDxCIndex) {
              // bottomRight
              accumulator += getDy(b, dyR, dyC, d) * dxRLerp * dxCLerp;
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function uq(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s,a=new cq(r.shape,o.shape,i);return e.runWebGLProgram(a,[r],r.dtype)}const hq={kernelName:Mh,backendName:"webgl",kernelFunc:uq};class dq{constructor(t,e,s,o,r){this.variableNames=["A"],this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s],d=o?"0.5":"0.0";let p;r?p="max((vec2(yRC) + vec2(0.5)) * effectiveInputOverOutputRatioRC, vec2(0.0))":p="vec2(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec2 effectiveInputOverOutputRatioRC = vec2(
          ${u[0]/h[0]},
          ${u[1]/h[1]});
      const vec2 inputShapeRC = vec2(${a}.0, ${l}.0);

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        ivec2 yRC = coords.yz;

        // Fractional source index.
        vec2 sourceFracIndexRC = ${p};

        // Compute the coordinators of nearest neighbor point.
        ivec2 sourceNearestRC = ivec2(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));
        float newValue = getA(b, sourceNearestRC.x, sourceNearestRC.y, d);

        setOutput(newValue);
      }
    `}}class pq{constructor(t,e,s,o,r){this.variableNames=["A"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=[];const[i,a,l,c]=t;this.outputShape=[i,e,s,c];const u=[o&&e>1?a-1:a,o&&s>1?l-1:l],h=[o&&e>1?e-1:e,o&&s>1?s-1:s],d=o?"0.5":"0.0";let p;r?p="max((vec3(yRC) + vec3(0.5)) * effectiveInputOverOutputRatioRC, vec3(0.0))":p="vec3(yRC) * effectiveInputOverOutputRatioRC",this.userCode=`
      const vec3 effectiveInputOverOutputRatioRC = vec3(
          ${u[0]/h[0]},
          ${u[1]/h[1]},
          ${u[1]/h[1]});
      const vec3 inputShapeRC = vec3(${a}.0, ${l}.0,
                                     ${l}.0);

      float getAValue(int b, int r, int c, int d) {
        return getChannel(getA(b, r, c, d), vec2(c, d));
      }

      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        // Calculate values for next column in yRC.z.
        ivec3 yRC = coords.yzz + ivec3(0, 0, 1);

        // Fractional source index.
        vec3 sourceFracIndexRC = ${p};

        // Compute the coordinators of nearest neighbor point.
        ivec3 sourceNearestRC = ivec3(
          min(inputShapeRC - 1.0, floor(sourceFracIndexRC + ${d})));

        // Should we calculate next column and row elements in 2x2 packed cell.
        bool hasNextCol = d < ${c-1};
        bool hasNextRow = coords.z < ${s-1};

        vec4 newValue = vec4(
          getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d),
          hasNextCol ? getAValue(b, sourceNearestRC.x, sourceNearestRC.y, d + 1)
                     : 0.0,
          hasNextRow ? getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d)
                     : 0.0,
          (hasNextRow && hasNextCol) ?
            getAValue(b, sourceNearestRC.x, sourceNearestRC.z, d + 1) : 0.0);

        setOutput(newValue);
      }
    `}}function fq(n){const{inputs:t,backend:e,attrs:s}=n,{images:o}=t,{alignCorners:r,halfPixelCenters:i,size:a}=s,[l,c]=a,u=B().getBool("WEBGL_PACK_IMAGE_OPERATIONS")?new pq(o.shape,l,c,r,i):new dq(o.shape,l,c,r,i);return e.runWebGLProgram(u,[o],o.dtype)}const mq={kernelName:oc,backendName:"webgl",kernelFunc:fq};class gq{constructor(t,e,s){this.variableNames=["dy"],this.outputShape=[],this.outputShape=e;const[,o,r]=e,[,i,a]=t,l=[s&&i>1?o-1:o,s&&a>1?r-1:r],c=[s&&i>1?i-1:i,s&&a>1?a-1:a],u=l[0]/c[0],h=l[1]/c[1],d=1/u,p=1/h,f=Math.ceil(d)*2+2,m=Math.ceil(p)*2+2;this.userCode=`
      void main() {
        ivec4 coords = getOutputCoords();
        int b = coords[0];
        int d = coords[3];
        int r = coords[1];
        int c = coords[2];

        float accumulator = 0.0;

        const float heightScale = float(${u});
        const float widthScale = float(${h});

        const float invHeightScale = float(${d});
        const float invWidthScale = float(${p});

        const int winHeight = int(${f});
        const int winWidth = int(${m});

        // Compute bounds for where in dy we will look
        float startRLerp = floor(float(r) * invHeightScale);
        int startDyR = int(floor(startRLerp - float(winHeight / 2)));

        float startCLerp = floor(float(c) * invWidthScale);
        int startDyC = int(floor(startCLerp - float(winWidth / 2)));

        // Loop over dy
        for (int dyROffset = 0; dyROffset < winHeight; dyROffset++) {
          int dyR = dyROffset + startDyR;

          // Guard against the window exceeding the bounds of dy
          if (dyR < 0 || dyR >= ${i}) {
            continue;
          }

          for (int dyCOffset = 0; dyCOffset < winWidth; dyCOffset++) {
            int dyC = dyCOffset + startDyC;

            // Guard against the window exceeding the bounds of dy
            if (dyC < 0 || dyC >= ${a}) {
              continue;
            }

            float sourceFracRow =
              float(${l[0]}) *
                (float(dyR) / float(${c[0]}));

            float sourceFracCol =
                float(${l[1]}) *
                  (float(dyC) / float(${c[1]}));

            int sourceNearestRow = int(min(
                float(int(${o}) - 1),
                ${s} ? float(round(sourceFracRow)) :
                                  float(floor(sourceFracRow))));

            int sourceNearestCol = int(min(
                float(int(${r}) - 1),
                ${s} ? float(round(sourceFracCol)) :
                                  float(floor(sourceFracCol))));

            if (r == sourceNearestRow && c == sourceNearestCol) {
              accumulator += getDy(b, dyR, dyC, d);
            }
          }
        }
        // End loop over dy

        setOutput(accumulator);
      }
    `}}function xq(n){const{inputs:t,backend:e,attrs:s}=n,{images:o,dy:r}=t,{alignCorners:i}=s,a=new gq(r.shape,o.shape,i);return e.runWebGLProgram(a,[r],r.dtype)}const bq={kernelName:Lh,backendName:"webgl",kernelFunc:xq};class yq{constructor(t,e){this.variableNames=["x"];const s=t.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);if(this.outputShape=t,s===1){this.userCode=`
        void main() {
          int coord = getOutputCoords();
          setOutput(getX(${t[0]} - coord - 1));
        }
      `;return}const o=a=>e.indexOf(a)!==-1&&t[a]!==1?`${t[a]} - coords[${a}] - 1`:`coords[${a}]`,r=t.map((a,l)=>o(l)).join(","),i=Ot(s);this.userCode=`
      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${r}));
      }
    `}}class wq{constructor(t,e){this.variableNames=["x"],this.packedInputs=!0,this.packedOutput=!0;const s=t.length;if(s>4)throw new Error(`WebGL backend: Reverse of rank-${s} tensor is not yet supported`);this.outputShape=t;const o=Fe("rc",s),r=`${o[s-1]} + 1 < ${this.outputShape[s-1]}`,i=`${o[s-2]} + 1 < ${this.outputShape[s-2]}`,a=Ot(s);s===1?this.userCode=`
        void main(){
          int rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = getChannel(getX(${t[0]} - rc - 1),
            ${t[0]} - rc - 1);
          if(${r}){
              result.g = getChannel(getX(${t[0]} - (rc  + 1) - 1),
                ${t[0]} - (rc  + 1) - 1);
          }
          setOutput(result);
        }
      `:this.userCode=`
        void main() {
          ${a} rc = getOutputCoords();
          vec4 result = vec4(0.);
          result.r = ${l(o.slice())};
          if(${r}){
            result.g = ${c(o.slice())};
          }
          if(${i}) {
            result.b = ${u(o.slice())};
            if(${r}) {
              result.a = ${h(o.slice())};
            }
          }
          setOutput(result);
        }
    `;function l(f){return d(f)}function c(f){return f[s-1]="("+f[s-1]+" + 1)",d(f)}function u(f){return f[s-2]="("+f[s-2]+" + 1)",d(f)}function h(f){return f[s-1]="("+f[s-1]+" + 1)",f[s-2]="("+f[s-2]+" + 1)",d(f)}function d(f){const m=t.map((b,w)=>p(w,f)),g=m.join(","),x=m.slice(-2).join(",");return`getChannel(getX(${g}), vec2(${x}))`}function p(f,m){return e.indexOf(f)!==-1&&t[f]!==1?`${t[f]} - ${m[f]} - 1`:`${m[f]}`}}}function $q(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{dims:r}=s,i=o.shape.length,a=$t(r,o.shape);if(i===0)return Ye({inputs:{x:o},backend:e});const l=B().getBool("WEBGL_PACK_ARRAY_OPERATIONS")?new wq(o.shape,a):new yq(o.shape,a);return e.runWebGLProgram(l,[o],o.dtype)}const Cq={kernelName:ic,backendName:"webgl",kernelFunc:$q};class Iq{constructor(t,e){this.variableNames=["Image"],this.outputShape=[],this.customUniforms=[{name:"params",type:"vec4"}];const s=t[1],o=t[2];this.outputShape=t;let r="";typeof e=="number"?r=`float outputValue = ${e.toFixed(2)};`:r=`
        vec3 fill = vec3(${e.join(",")});
        float outputValue = fill[coords[3]];`,this.userCode=`
        void main() {
          ivec4 coords = getOutputCoords();
          int x = coords[2];
          int y = coords[1];
          float coordXFloat = (float(x) - params[0]) * params[3] -
            (float(y) - params[1]) * params[2];
          float coordYFloat = (float(x) - params[0]) * params[2] +
            (float(y) - params[1]) * params[3];
          int coordX = int(round(coordXFloat + params[0]));
          int coordY = int(round(coordYFloat + params[1]));
          ${r}
          if(coordX >= 0 && coordX < ${o} && coordY >= 0 && coordY < ${s}) {
            outputValue = getImage(coords[0], coordY, coordX, coords[3]);
          }
          setOutput(outputValue);
        }
    `}}const vq={kernelName:Yh,backendName:"webgl",kernelFunc:({inputs:n,attrs:t,backend:e})=>{const{image:s}=n,{radians:o,fillValue:r,center:i}=t,a=e,l=new Iq(s.shape,r),[c,u]=Kd(i,s.shape[1],s.shape[2]),h=[[c,u,Math.sin(o),Math.cos(o)]];return a.runWebGLProgram(l,[s],s.dtype,h)}};const kq=`
  // OpenGL ES does not support round function.
  // The algorithm is based on banker's rounding.
  float base = floor(x);
  if ((x - base) < 0.5) {
    return floor(x);
  } else if ((x - base) > 0.5) {
    return ceil(x);
  } else {
    if (mod(base, 2.0) == 0.0) {
      return base;
    } else {
      return base + 1.0;
    }
  }
`,Sq=St({opSnippet:kq}),Nq={kernelName:Ei,backendName:"webgl",kernelFunc:Sq};const Tq="return inversesqrt(x);",Eq=St({opSnippet:Tq,cpuKernelImpl:Vz}),Rq={kernelName:Ri,backendName:"webgl",kernelFunc:Eq};class of{constructor(t,e,s,o,r,i,a=!0,l=!1){this.variableNames=["updates","indices","defaultValue"],this.outputShape=i;const c=Ot(r.length),u=Ot(i.length);let h="";s===1?h="i":s===2&&(h="i, j");const d=`getIndices(${h})`;let p="";o===1?p="i":o===2&&(p="i, coords[1]");const f=`getUpdates(${p})`;let m="";l&&(m="coords[0], coords[1]");const g=`getDefaultValue(${m})`,x=e>1?"strides[j]":"strides";this.userCode=`
        ${c} strides = ${c}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          float sum = 0.0;
          bool found = false;
          for (int i = 0; i < ${t}; i++) {
            int flattenedIndex = 0;
            for (int j = 0; j < ${e}; j++) {
              int index = round(${d});
              flattenedIndex += index * ${x};
            }
            if (flattenedIndex == coords[0]) {
              sum += ${f};
              found = true;
            }
          }
          setOutput(mix(${g}, sum, float(found)));
        }
      `}}class Aq{constructor(t,e,s,o,r,i,a=!0,l=!1){this.variableNames=["updates","indices","defaultValue"],this.packedInputs=!0,this.packedOutput=!0,this.outputShape=i;const c=Ot(r.length),u=Ot(i.length);let h="";s===1?h="i":s===2&&(h="i, j");const d=`getIndices(${h})`;let p="";o===1?p="i":o===2&&(p="i, coords[1]");const f=`getUpdates(${p})`;let m="";l&&(m="coords[0], coords[1]");const g=`getDefaultValue(${m})`,x=e>1?"strides[j]":"strides",b=e>1?"strides[j + 1]":"strides";this.userCode=`
        ${c} strides = ${c}(${r});

        void main() {
          ${u} coords = getOutputCoords();
          vec4 sum = vec4(0.);
          vec4 found = vec4(0.);
          for (int i = 0; i < ${t}; i+=2) {
            ivec2 flattenedIndex = ivec2(0);
            for (int j = 0; j < ${e}; j+=2) {
              ivec4 index = round(${d});
              flattenedIndex += index.xz * ${x};
              if (j + 1 < ${e}) {
                flattenedIndex += index.yw * ${b};
              }
            }
            if (flattenedIndex[0] == coords[0] || flattenedIndex[1] == coords[0] ||
                flattenedIndex[0] == coords[0] + 1 || flattenedIndex[1] == coords[0] + 1) {
              vec4 updVals = ${f};
              if (flattenedIndex[0] == coords[0]) {
                sum.xy += updVals.xy;
                found.xy = vec2(1.);
              } else if (flattenedIndex[0] == coords[0] + 1) {
                sum.zw += updVals.xy;
                found.zw = vec2(1.);
              }
              if (flattenedIndex[1] == coords[0]) {
                sum.xy += updVals.zw;
                found.xy = vec2(1.);
              } else if (flattenedIndex[1] == coords[0] + 1) {
                sum.zw += updVals.zw;
                found.zw = vec2(1.);
              }
            }
          }
          setOutput(mix(${g}, sum, found));
        }
      `}}function Dq(n){const{inputs:t,backend:e,attrs:s}=n,{indices:o,updates:r}=t,{shape:i}=s,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Bs(r,o,i),d=[h/c,c];if(h===0)return e.makeTensorInfo(i,o.dtype);const p=et({inputs:{x:o},backend:e,attrs:{shape:[l,a]}}),f=et({inputs:{x:r},backend:e,attrs:{shape:[l,c]}}),m=e.makeTensorInfo([],"float32",new Float32Array([0]));let g;B().getBool("WEBGL_PACK")?g=new Aq(l,a,p.shape.length,f.shape.length,u,d):g=new of(l,a,p.shape.length,f.shape.length,u,d);const x=e.runWebGLProgram(g,[f,p,m],f.dtype),b=et({inputs:{x},backend:e,attrs:{shape:i}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(x),e.disposeIntermediateTensorInfo(m),b}const Fq={kernelName:lg,backendName:"webgl",kernelFunc:Dq};class _q{constructor(t,e,s,o){this.variableNames=["sortedSequence","values"],this.customUniforms=[{name:"numInputs",type:"int"}],this.outputShape=[t,s];const r="while (left < right) {",i=`for (int i = 0; i < ${Math.ceil(Math.log2(e+1))}; ++i) { if (left >= right) break;`,a=B().getNumber("WEBGL_VERSION")===2?r:i,l=o==="left"?"<":"<=";this.userCode=`
       int findBound(int batch, float value) {
         int left = 0;
         int right = numInputs;
         int mid;
         ${a}
           mid = (left + right) / 2;
           if (getSortedSequence(batch, mid) ${l} value) {
             left = mid + 1;
           } else {
             right = mid;
           }
         }
         return right;
       }

       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int valueIndex = coords[1];

         float value = getValues(batch, valueIndex);

         setOutput(float(findBound(batch, value)));
       }
     `}}function Oq(n){const{inputs:t,backend:e,attrs:s}=n,{sortedSequence:o,values:r}=t,{side:i}=s,a=new _q(o.shape[0],o.shape[1],r.shape[1],i),l=[[o.shape[1]]];return e.runWebGLProgram(a,[o,r],"int32",l)}const Lq={kernelName:ug,backendName:"webgl",kernelFunc:Oq};class Mq{constructor(t,e,s){this.variableNames=["c","a","b"],this.outputShape=e;let o,r;if(s>4)throw Error(`Where for rank ${s} is not yet supported`);if(s===1)r="resRC",o="resRC";else{const a=["resRC.x","resRC.y","resRC.z","resRC.w"],l=[],c=[];for(let u=0;u<e.length;u++)c.push(`${a[u]}`),u<t&&l.push(`${a[u]}`);o=l.join(),r=c.join()}const i=Ot(s);this.userCode=`
      void main() {
        ${i} resRC = getOutputCoords();
        float cVal = getC(${o});
        if (cVal >= 1.0) {
          setOutput(getA(${r}));
        } else {
          setOutput(getB(${r}));
        }
      }
    `}}function Pq(n){const{inputs:t,backend:e}=n,{condition:s,t:o,e:r}=t,i=new Mq(s.shape.length,o.shape,o.shape.length);return e.runWebGLProgram(i,[s,o,r],Ke(o.dtype,r.dtype))}const zq={kernelName:ac,backendName:"webgl",kernelFunc:Pq};const Bq=`
  // Stable and Attracting Fixed Point (0, 1) for Normalized Weights.
  // see: https://arxiv.org/abs/1706.02515
  float scaleAlpha = ${Ec};
  float scale = ${Rc};
  return (x >= 0.0) ? scale * x : scaleAlpha * (exp(x) - 1.0);
`,Vq=St({opSnippet:Bq}),Wq={kernelName:Ai,backendName:"webgl",kernelFunc:Vq};const Uq=pr+`
  return 1.0 / (1.0 + exp(-1.0 * x));
`,Gq=`
  vec4 result = 1.0 / (1.0 + exp(-1.0 * x));
  bvec4 isNaN = isnan(x);

  result.r = isNaN.r ? x.r : result.r;
  result.g = isNaN.g ? x.g : result.g;
  result.b = isNaN.b ? x.b : result.b;
  result.a = isNaN.a ? x.a : result.a;

  return result;
`,Hq=St({opSnippet:Uq,packedOpSnippet:Gq,cpuKernelImpl:Uz}),qq={kernelName:Oi,backendName:"webgl",kernelFunc:Hq};const jq=`
  if (isnan(x)) { return 0.0; }
  return sign(x);
`,Kq=St({opSnippet:jq}),Xq={kernelName:_i,backendName:"webgl",kernelFunc:Kq};const Yq=pr+`
  return sin(x);
`,Zq=`
  vec4 result = sin(x);
  bvec4 isNaN = isnan(x);
  ${Do}
  return result;
`,Jq=St({opSnippet:Yq,packedOpSnippet:Zq}),Qq={kernelName:Di,backendName:"webgl",kernelFunc:Jq};const tj=`
  float e2x = exp(x);
  return (e2x - 1.0 / e2x) / 2.0;
`,ej=St({opSnippet:tj}),nj={kernelName:Fi,backendName:"webgl",kernelFunc:ej};const sj=`
  float epsilon = 1.1920928955078125e-7;
  float threshold = log(epsilon) + 2.0;

  bool too_large = x > -threshold;
  bool too_small = x < threshold;

  float result;
  float exp_x = exp(x);

  if (too_large){
    result = x;
  }
  else if (too_small){
    result = exp_x;
  }
  else{
    result = log(exp_x + 1.0);
  }
  return result;
`,oj=St({opSnippet:sj}),rj={kernelName:Li,backendName:"webgl",kernelFunc:oj};const ij=n=>{const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{blockShape:r,paddings:i}=s;I(o.shape.length<=4,()=>"spaceToBatchND for rank > 4 with a WebGL backend not implemented yet");const a=r.reduce((x,b)=>x*b),l=[[0,0]];l.push(...i);for(let x=1+r.length;x<o.shape.length;++x)l.push([0,0]);const c=[],u=r$({inputs:{x:o},backend:e,attrs:{paddings:l,constantValue:0}}),h=Zi(u.shape,r,a,!1),d=Ji(h.length,r.length,!1),p=Qi(u.shape,r,a,!1),f=et({inputs:{x:u},backend:e,attrs:{shape:h}}),m=_e({inputs:{x:f},backend:e,attrs:{perm:d}}),g=et({inputs:{x:m},backend:e,attrs:{shape:p}});return c.push(u),c.push(f),c.push(m),c.forEach(x=>e.disposeIntermediateTensorInfo(x)),g},aj={kernelName:uc,backendName:"webgl",kernelFunc:ij};function lj(n){const{inputs:t,backend:e}=n,{indices:s,values:o,denseShape:r,defaultValue:i}=t;if(r.shape.length!==1)throw new Error(`Dense shape must be a vector, saw:
         ${r.shape}`);if(s.shape.length!==2)throw new Error(`Indices must be a matrix, saw:
         ${s.shape}`);if(o.shape.length!==1)throw new Error(`Values must be a vector, saw:
         ${o.shape}`);if(i.shape.length!==0)throw new Error(`Default value must be a scalar, saw:
        ${i.shape}`);const a=e.readSync(s.dataId),l=e.readSync(o.dataId),c=e.readSync(r.dataId),u=e.readSync(i.dataId)[0],[h,d,p,f,m]=Hz(a,s.shape,s.dtype,l,o.dtype,c,u);return[e.makeTensorInfo(d,s.dtype,h),e.makeTensorInfo([d[0]],o.dtype,p),e.makeTensorInfo([f.length],"bool",new Uint8Array(f.map(g=>Number(g)))),e.makeTensorInfo([m.length],s.dtype,new Int32Array(m))]}const cj={kernelName:Ph,backendName:"webgl",kernelFunc:lj};function uj(n){const{inputs:t,backend:e}=n,{inputIndices:s,inputShape:o,newShape:r}=t;if(s.shape.length!==2)throw new Error(`Input indices should be a matrix but received shape ${s.shape}`);if(o.shape.length!==1)throw new Error(`Input shape should be a vector but received shape ${o.shape}`);if(r.shape.length!==1)throw new Error(`Target shape should be a vector but received shape ${r.shape}`);const i=Array.from(e.readSync(o.dataId)),a=e.readSync(s.dataId),l=Array.from(e.readSync(r.dataId)),[c,u,h]=qz(a,s.shape,s.dtype,i,l);return[e.makeTensorInfo(u,s.dtype,c),e.makeTensorInfo([h.length],r.dtype,new Int32Array(h))]}const hj={kernelName:zh,backendName:"webgl",kernelFunc:uj};function dj(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
              ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
              ${r.shape}`);const i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=Rw(i,s.shape,s.dtype,a,l,!0);return e.makeTensorInfo(u,s.dtype,c)}const pj={kernelName:Bh,backendName:"webgl",kernelFunc:dj};function fj(n){const{inputs:t,backend:e}=n,{data:s,indices:o,segmentIds:r}=t;if(s.shape.length<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(o.shape.length!==1)throw new Error(`Indices should be a vector but received shape
             ${o.shape}`);if(r.shape.length!==1)throw new Error(`Segment ids should be a vector but received shape
             ${r.shape}`);const i=e.readSync(s.dataId),a=e.readSync(o.dataId),l=e.readSync(r.dataId),[c,u]=Rw(i,s.shape,s.dtype,a,l);return e.makeTensorInfo(u,s.dtype,c)}const mj={kernelName:Vh,backendName:"webgl",kernelFunc:fj};function gj(n){const{inputs:t,backend:e,attrs:s}=n,{sparseIndices:o,sparseValues:r,defaultValue:i}=t,{outputShape:a}=s,{sliceRank:l,numUpdates:c,sliceSize:u,strides:h,outputSize:d}=Bs(r,o,a),p=!1;if(r.dtype==="string"){const x=e.bufferSync(o),b=e.bufferSync(r),w=ns(e.readSync(i.dataId)[0]),y=Wz(x,b,a,d,u,c,l,h,w,p);return e.makeTensorInfo(a,y.dtype,y.values)}const f=new of(c,l,o.shape.length,r.shape.length,h,[d,1],p),m=e.runWebGLProgram(f,[r,o,i],r.dtype),g=et({inputs:{x:m},backend:e,attrs:{shape:a}});return e.disposeIntermediateTensorInfo(m),g}const xj={kernelName:hg,backendName:"webgl",kernelFunc:gj};function bj(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{numOrSizeSplits:r,axis:i}=s,a=$t(i,o.shape)[0],l=cp(o,r,a),c=o.shape.length,u=new Array(c).fill(0),h=o.shape.slice();return l.map(d=>{const p=[...h];p[a]=d;const f=fr({inputs:{x:o},backend:e,attrs:{begin:u,size:p}});return u[a]+=d,f})}const yj={kernelName:hc,backendName:"webgl",kernelFunc:bj};const Vm="return sqrt(x);",wj=St({opSnippet:Vm,packedOpSnippet:Vm,cpuKernelImpl:jz}),$j={kernelName:Mi,backendName:"webgl",kernelFunc:wj};const Cj="return x * x;",Ij=St({opSnippet:Cj}),vj={kernelName:Wh,backendName:"webgl",kernelFunc:Ij};const Wm="return (a - b) * (a - b);",kj=ve({opSnippet:Wm,packedOpSnippet:Wm}),Sj={kernelName:Pi,backendName:"webgl",kernelFunc:kj};function Nj(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t;if(o.dtype!=="string")throw new Error("Input must be of datatype string");const r=e.readSync(o.dataId),i=rs(r),a=Kz(i,"string",s);return e.makeTensorInfo(o.shape,"string",a)}const Tj={kernelName:pc,backendName:"webgl",kernelFunc:Nj};function Ej({inputs:n,attrs:t,backend:e}){const{x:s}=n,o=yn+`
    return x > 0.0 ? 1.0 : float(${t.alpha});
  `,r=new zn(s.shape,o);return e.runWebGLProgram(r,[s],s.dtype)}const Rj={kernelName:Ui,backendName:"webgl",kernelFunc:Ej};class Aj{constructor(t,e,s){this.variableNames=["x"],this.outputShape=s;const o=s.length,r=Ot(s.length),i=Ot(s.length);let a="";if(o===1)a="coords * strides + begin";else{let l=0;a=s.map((c,u)=>(l++,s.length===1?`coords * strides[${u}] + begin[${u}]`:`coords[${l-1}] * strides[${u}] + begin[${u}]`)).join(",")}this.userCode=`
      ${r} begin = ${r}(${t});
      ${r} strides = ${r}(${e});

      void main() {
        ${i} coords = getOutputCoords();
        setOutput(getX(${a}));
      }
    `}}function Dj(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{begin:r,end:i,strides:a,beginMask:l,endMask:c,ellipsisMask:u,newAxisMask:h,shrinkAxisMask:d}=s,{finalShapeSparse:p,finalShape:f,isIdentity:m,sliceDim0:g,isSimpleSlice:x,begin:b,end:w,strides:y}=Hd(o.shape,r,i,a,l,c,u,h,d);let $;if(m)$=et({inputs:{x:o},backend:e,attrs:{shape:f}});else if(g||x){I(o.shape.length>=1,()=>`Input must have rank at least 1, got: ${o.shape.length}`);const N=Wd(b,w,y),T=fr({inputs:{x:o},backend:e,attrs:{begin:b,size:N}});$=et({inputs:{x:T},backend:e,attrs:{shape:f}}),e.disposeIntermediateTensorInfo(T)}else if(e.shouldExecuteOnCPU([o])){const T=e.readSync(o.dataId),k=Ct(o.shape,o.dtype,T),S=Xz(p,k,y,b);$=e.makeTensorInfo(f,o.dtype,S.values)}else{const T=new Aj(b,y,p);$=e.runWebGLProgram(T,[o],o.dtype)}const v=et({inputs:{x:$},backend:e,attrs:{shape:f}});return e.disposeIntermediateTensorInfo($),v}const Fj={kernelName:Uh,backendName:"webgl",kernelFunc:Dj};function _j(n){const{inputs:t,backend:e,attrs:s}=n,{separator:o,nGramWidths:r,leftPad:i,rightPad:a,padWidth:l,preserveShortSequences:c}=s,{data:u,dataSplits:h}=t,d=e.readSync(u.dataId),p=e.readSync(h.dataId),[f,m]=Yz(d,p,o,r,i,a,l,c);return[e.makeTensorInfo([f.length],"string",f),e.makeTensorInfo(h.shape,"int32",m)]}const Oj={kernelName:Gh,backendName:"webgl",kernelFunc:_j};function Lj(n){const{inputs:t,backend:e,attrs:s}=n,{skipEmpty:o}=s,{input:r,delimiter:i}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(r.shape.length!==1)throw new Error(`Input must be a vector, got shape: ${r.shape}`);if(i.shape.length!==0)throw new Error(`Delimiter must be a scalar, got shape: ${i.shape}`);const a=e.readSync(r.dataId),l=e.readSync(i.dataId)[0],[c,u,h]=Zz(a,l,o),d=u.length;return[e.makeTensorInfo([d,2],"int32",c),e.makeTensorInfo([d],"string",u),e.makeTensorInfo([2],"int32",new Int32Array(h))]}const Mj={kernelName:Hh,backendName:"webgl",kernelFunc:Lj};function Pj(n){const{inputs:t,backend:e,attrs:s}=n,{numBuckets:o}=s,{input:r}=t;if(r.dtype!=="string")throw new Error("Input must be of datatype string");if(o<=0)throw new Error("Number of buckets must be at least 1");const i=e.readSync(r.dataId),a=Jz(i,o);return e.makeTensorInfo(r.shape,"int32",a)}const zj={kernelName:qh,backendName:"webgl",kernelFunc:Pj};const Bj="return tan(x);",Vj=St({opSnippet:Bj}),Wj={kernelName:Bi,backendName:"webgl",kernelFunc:Vj};const Uj=`
  float e2x = exp(-2.0 * abs(x));
  return sign(x) * (1.0 - e2x) / (1.0 + e2x);
`,Gj=St({opSnippet:Uj}),Hj={kernelName:Vi,backendName:"webgl",kernelFunc:Gj};function qj(n){const{inputs:t,backend:e,attrs:s}=n,{tensor:o,indices:r,updates:i}=t,{sliceRank:a,numUpdates:l,sliceSize:c,strides:u,outputSize:h}=Bs(i,r,o.shape),d=[h/c,c];if(h===0)return e.makeTensorInfo(o.shape,r.dtype);const p=et({inputs:{x:r},backend:e,attrs:{shape:[l,a]}}),f=et({inputs:{x:i},backend:e,attrs:{shape:[l,c]}}),m=et({inputs:{x:o},backend:e,attrs:{shape:d}}),g=new of(l,a,p.shape.length,f.shape.length,u,d,!1,!0),x=e.runWebGLProgram(g,[f,p,m],m.dtype),b=et({inputs:{x},backend:e,attrs:{shape:o.shape}});return e.disposeIntermediateTensorInfo(p),e.disposeIntermediateTensorInfo(f),e.disposeIntermediateTensorInfo(m),e.disposeIntermediateTensorInfo(x),b}const jj={kernelName:cg,backendName:"webgl",kernelFunc:qj};class Kj{constructor(t,e){this.variableNames=["A"];const s=new Array(t.length);for(let i=0;i<s.length;i++)s[i]=t[i]*e[i];this.outputShape=s,this.rank=s.length;const o=Ot(this.rank),r=Xj(t);this.userCode=`
      void main() {
        ${o} resRC = getOutputCoords();
        setOutput(getA(${r}));
      }
    `}}function Xj(n){const t=n.length;if(t>5)throw Error(`Tile for rank ${t} is not yet supported`);if(t===1)return`imod(resRC, ${n[0]})`;const e=["resRC.x","resRC.y","resRC.z","resRC.w","resRC.u"],s=[];for(let o=0;o<n.length;o++)s.push(`imod(${e[o]}, ${n[o]})`);return s.join()}function a$(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{reps:r}=s;if(o.dtype==="string"||o.shape.length>5){const l=e.readSync(o.dataId),c=o.dtype==="string"?l.map(d=>ns(d)):l,u=Ct(o.shape,o.dtype,c),h=tB(u,r);return e.makeTensorInfo(h.shape,h.dtype,h.values)}const i=new Kj(o.shape,r);return e.runWebGLProgram(i,[o],o.dtype)}const Yj={kernelName:Wi,backendName:"webgl",kernelFunc:a$};class Zj{constructor(t){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"negativeInf",type:"float"},{name:"dir",type:"int"},{name:"inc",type:"int"}],this.outputShape=t,this.userCode=`
       void main() {
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // We compare elements pair-wise within a group of size 2 * inc.
         // The comparing rule for each group alternates between ascending
         // and descending. Within each group, we compare each pair at
         // positions i and i+inc. To decide whether an element at position i
         // is x0 or x1, we mod it by 2 * inc, if the result is smaller than
         // inc, it is in the first half of the group, we denote it as x0,
         // otherwise we denote it as x1.
         // For example, as shown in the Bitonic top K paper referenced above,
         // Figure5(a) shows that element[1] is in the
         // second half of the group when group size is 2, but it is in the
         // first half of the group when group size is 4.

         bool isFirstInPair = imod(elemIdx, 2 * inc) < inc;
         int i = isFirstInPair ? elemIdx : elemIdx - inc;

         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + inc : int(getIndices(batch, i + inc));
         float x0 = i0 < n ? getX(batch, i0) : negativeInf;
         float x1 = i1 < n ? getX(batch, i1) : negativeInf;

         // Denotes which direction indices are in (ascending or descending).
         bool reverse = imod(elemIdx, 2 * dir) >= dir;
         bool isGreater = x0 > x1 || (x0 == x1 && i1 > i0);
         if (reverse == isGreater) { // Elements in opposite order of direction
           int iTemp = i0;
           i0 = i1;
           i1 = iTemp;
         }
         if (isFirstInPair) {
            setOutput(float(i0));
         } else {
            setOutput(float(i1));
         }
       }
     `}}class Jj{constructor(t){this.variableNames=["x","indices"],this.customUniforms=[{name:"n",type:"int"},{name:"firstPass",type:"int"},{name:"k",type:"int"}],this.outputShape=t,this.userCode=`
    void main() {
         // Takes max of indices (0, k), (1, k + 1), (2, k + 2) ...
         ivec2 coords = getOutputCoords();
         int batch = coords[0];
         int elemIdx = coords[1];

         // The output size is half of the previous size.
         // If the previous sequence is | | | | _ _ _ _  | | | |  _ _ _ _ (k=4),
         // we only need to output the indices at positions |, the indices at
         // positions _ can be thrown away, see Figure5(b) After Phase 2
         // (Merge phase) in the Bitonic Top K paper referenced above.
         // For example, the paper shows we only need to output the orange bars.
         // The output sequence should look like this | | | | | | | |.
         // Because the sequence is halved, to map the output index back
         // to the previous sequence to find the corresponding value,
         // we need to double the index. When we double the index,
         // we basically interpolate a position, so 2i looks like
         // | _ | _ | _ | _ | _ | _ | _. We move the | to the first k position
         // of each 2k positions by - elemIdx % k. E.g. for output at
         // index 4,5,6,7, we want to get the corresponding element at
         // original index 8,9,10,11, for output at index 8,9,10,11,
         // we want to get the corresponding element at original index
         // 16,17,18,19, so on and so forth.

         int i = elemIdx < k ? elemIdx : (elemIdx * 2 - imod(elemIdx, k));
         int i0 = firstPass == 1 ? i : int(getIndices(batch, i));
         int i1 = firstPass == 1 ? i + k : int(getIndices(batch, i + k));

         float x0 = getX(batch, i0);
         float x1 = i1 < n ? getX(batch, i1) : x0;

         setOutput(x0 >= x1 ? float(i0) : float(i1));
       }
     `}}function js(n,t){t!==null&&n.disposeIntermediateTensorInfo(t)}function Um(n){let t=1;for(;t<n;)t*=2;return t}function Qj(n){const{inputs:t,backend:e,attrs:s}=n,{x:o}=t,{k:r,sorted:i}=s,a=B().getNumber("TOPK_LAST_DIM_CPU_HANDOFF_SIZE_THRESHOLD"),l=B().getNumber("TOPK_K_CPU_HANDOFF_THRESHOLD"),c=o.shape,u=c[c.length-1];if(e.shouldExecuteOnCPU([o])||u<a||r>l){const S=e.readSync(o.dataId),[C,D]=eB(S,c,o.dtype,r,i);return[e.makeTensorInfo(C.shape,C.dtype,C.values),e.makeTensorInfo(D.shape,D.dtype,D.values)]}if(r===0)return c[c.length-1]=0,[e.makeTensorInfo(c,o.dtype,[]),e.makeTensorInfo(c,"int32",[])];if(u===1)return[o,da({attrs:{shape:c,dtype:"int32",value:0},backend:e})];const h=e.texData.get(o.dataId),d=h!==null&&h.isPacked,p=d?e.unpackTensor(o):o,m=U(c)/u,g=et({inputs:{x:p},attrs:{shape:[m,u]},backend:e});d&&js(e,p);const x=Um(r),b=Um(u);let w=null;const y=()=>w===null?[g,g]:[g,w],$=(S,C,D)=>{const O=y(),P=new Zj(D),V=[[u],[w===null?1:0],[Number.NEGATIVE_INFINITY],[S],[C]],H=w;w=e.runWebGLProgram(P,O,"int32",V),js(e,H)};for(let S=1;S<x;S*=2){const C=S*2;for(let D=S;D>=1;D/=2)$(C,D,[m,b])}for(let S=b;S>x;S/=2){const C=y(),D=new Jj([m,S/2]),P=[[u],[w===null?1:0],[x]],M=w;w=e.runWebGLProgram(D,C,"int32",P),js(e,M);const V=x/2,H=V*2;for(let W=V;W>=1;W/=2)$(H,W,w.shape)}let v=w;w=fr({inputs:{x:w},backend:e,attrs:{begin:0,size:[m,r]}}),js(e,v);let N=Qw({inputs:{x:g,indices:w},backend:e,attrs:{axis:1,batchDims:1}});js(e,g);const T=c.slice(0,-1);T.push(r),v=w,w=et({inputs:{x:w},attrs:{shape:T},backend:e}),js(e,v);const k=N;return N=et({inputs:{x:N},attrs:{shape:T},backend:e}),js(e,k),[N,w]}const tK={kernelName:jh,backendName:"webgl",kernelFunc:Qj};class eK{constructor(t,e,s,o,r,i){this.variableNames=["Image","Transforms"],this.outputShape=i;const a=s==="nearest"?1:2;let l;switch(o){case"constant":l=1;break;case"reflect":l=2;break;case"wrap":l=3;break;case"nearest":l=4;break;default:l=1;break}this.userCode=`
            float mapCoord(float outCoord, float len) {
              float inCoord = outCoord;
              if(${l} == 2) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    if (inCoord < sz2) {
                      inCoord = sz2 * float(int(float(-inCoord / sz2))) +
                      inCoord;
                    }
                    inCoord = inCoord < -len ? inCoord + sz2 : -inCoord - 1.0;
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz2 = 2.0 * len;
                    inCoord -= sz2 * float(int(float(inCoord / sz2)));
                    if (inCoord >= len) {
                      inCoord = sz2 - inCoord - 1.0;
                    }
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${l} == 3) {
                if (inCoord < 0.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord += len * (float(int(float(-inCoord / sz))) + 1.0);
                  }
                } else if (inCoord > len - 1.0) {
                  if (len <= 1.0) {
                    inCoord = 0.0;
                  } else {
                    float sz = len - 1.0;
                    inCoord -= len * float(int(float(inCoord / sz)));
                  }
                }
                return clamp(inCoord, 0.0, len - 1.0);
              } else if (${l} == 4) {
                return clamp(outCoord, 0.0, len - 1.0);
              } else {
                return outCoord;
              }
            }

            float readWithFillValue(int batch, int coordY, int coordX,
              int channel) {
              float outputValue;
              if (0 <= coordY && coordY < ${t} && 0 <= coordX && coordX < ${e}) {
                  outputValue = getImage(batch, coordY, coordX, channel);
              } else {
                outputValue = float(${r});
              }
              return outputValue;
            }

            void main() {
              ivec4 coords = getOutputCoords();
              float outputValue;
              int batch = coords[0];
              int x = coords[2];
              int y = coords[1];
              int channel = coords[3];
              float xf = float(x);
              float yf = float(y);
              float a1 = getTransforms(batch, 0);
              float a2 = getTransforms(batch, 1);
              float a3 = getTransforms(batch, 2);
              float b1 = getTransforms(batch, 3);
              float b2 = getTransforms(batch, 4);
              float b3 = getTransforms(batch, 5);
              float c1 = getTransforms(batch, 6);
              float c2 = getTransforms(batch, 7);
              float projection = c1 * xf + c2 * yf + 1.0;
              if (projection == 0.0) {
                outputValue = float(${r});
              } else {
                float inX = (a1 * xf + a2 * yf + a3) / projection;
                float inY = (b1 * xf + b2 * yf + b3) / projection;
                float mapX = mapCoord(inX, float(${e}));
                float mapY = mapCoord(inY, float(${t}));

                if (${a} == 1) {
                  int coordY = int(round(mapY));
                  int coordX = int(round(mapX));
                  outputValue = readWithFillValue(batch, coordY, coordX,
                    channel);
                } else {
                  float yFloor = floor(mapY);
                  float xFloor = floor(mapX);
                  float yCeil = yFloor + 1.0;
                  float xCeil = xFloor + 1.0;
                  float valueYFloor = (xCeil - mapX) *
                  readWithFillValue(batch, int(yFloor), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yFloor), int(xCeil), channel);
                  float valueYCeil = (xCeil - mapX) *
                  readWithFillValue(batch, int(yCeil), int(xFloor), channel) +
                  (mapX - xFloor) *
                  readWithFillValue(batch, int(yCeil), int(xCeil), channel);
                  outputValue = (yCeil - mapY) * valueYFloor +
                  (mapY - yFloor) * valueYCeil;
                }
              }
              setOutput(outputValue);
            }
        `}}function nK(n){const{inputs:t,backend:e,attrs:s}=n,{image:o,transforms:r}=t,{interpolation:i,fillMode:a,fillValue:l,outputShape:c}=s,[u,h,d,p]=o.shape,[f,m]=c??[h,d],g=[u,f,m,p],x=new eK(h,d,i,a,l,g);return e.runWebGLProgram(x,[o,r],"float32")}const sK={kernelName:Kh,backendName:"webgl",kernelFunc:nK};function oK(n){const{inputs:t,attrs:e,backend:s}=n,{axis:o}=e,{x:r}=t;ar(r,"unique"),console.warn("WARNING: ","UI might be locked temporarily as data is being downloaded");const i=s.readSync(r.dataId),{outputValues:a,outputShape:l,indices:c}=nB(i,o,r.shape,r.dtype);return[s.makeTensorInfo(l,r.dtype,a),s.makeTensorInfo([c.length],"int32",c)]}const rK={kernelName:Xh,backendName:"webgl",kernelFunc:oK};function iK(n){const{inputs:t,backend:e,attrs:s}=n,{value:o}=t;let{axis:r}=s;r<0&&(r+=o.shape.length);const i=o,a=i.shape.length,l=o.shape[r],c=new Array(a-1);let u=0;for(let m=0;m<a;m++)m!==r&&(c[u++]=i.shape[m]);const h=[],d=new Array(a).fill(0),p=i.shape.slice();p[r]=1;const f=new Array(l);for(let m=0;m<f.length;m++){d[r]=m;const g=fr({inputs:{x:i},backend:e,attrs:{begin:d,size:p}}),x=et({inputs:{x:g},backend:e,attrs:{shape:c}});f[m]=x,h.push(g)}return h.forEach(m=>e.disposeIntermediateTensorInfo(m)),f}const aK={kernelName:fc,backendName:"webgl",kernelFunc:iK};class lK{constructor(t,e){this.variableNames=["x","segmentIds"];const s=t.windowSize,o=t.batchSize,r=t.inSize,i=t.numSegments,a=i*Math.ceil(r/s);this.outputShape=[o,a];const l="0.0",c="sumValue",u=Math.floor(s/4)*4,h=s%4,d=`
        sumValue += dot(values, segFilter);
    `;let p="";r%s>0&&(p=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return initializationValue;
        }
      `);let f="";r%s>0&&(f=`
        if (inIdx < 0 || inIdx >= ${r}) {
          return -1.0;
        }
      `),this.userCode=`
      const float initializationValue = ${l};

      float getValue(int batch, int inIdx) {
        ${p}
        return getX(batch, inIdx);
      }

      float getSegmentIdAtIndex(int inIdx) {
        ${f}
        return getSegmentIds(inIdx);
      }

      void main() {
        ivec2 coords = getOutputCoords();
        int batch = coords[0];
        int outIdx = coords[1];
        int inOffset = int(floor(float(outIdx) / float(
          ${i})) * float(${s}));
        int currentSeg = int(mod(float(outIdx), float(${i})));

        float sumValue = 0.0;

        for (int i = 0; i < ${u}; i += 4) {
          int inIdx = inOffset + i;
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            getValue(batch, inIdx + 3)
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 3)) == currentSeg ? 1 : 0
          );

          ${d}
        }

        int inIdx = inOffset + ${u};
        if (${h===1}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            initializationValue,
            initializationValue,
            initializationValue
          );

          int inIdxSeg = int(getSegmentIdAtIndex(inIdx));

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            0,
            0,
            0
          );

          ${d}
        } else if (${h===2}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            initializationValue,
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
              0,
              0
          );

          ${d}
        } else if (${h===3}) {
          vec4 values = vec4(
            getValue(batch, inIdx),
            getValue(batch, inIdx + 1),
            getValue(batch, inIdx + 2),
            initializationValue
          );

          vec4 segFilter = vec4(
            int(getSegmentIdAtIndex(inIdx)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 1)) == currentSeg ? 1 : 0,
            int(getSegmentIdAtIndex(inIdx + 2)) == currentSeg ? 1 : 0,
            0
          );

          ${d}
        }
        setOutput(${c});
      }
    `}}function cK(n){const{inputs:t,backend:e,attrs:s}=n,{x:o,segmentIds:r}=t,{numSegments:i}=s,a=o.shape.length,l=[];let c=0;const u=Xt([c],a);let h=o;u!=null&&(h=_e({inputs:{x:o},backend:e,attrs:{perm:u}}),l.push(h),c=ne(1,a)[0]);const d=eb(h.shape,c,i),p=U([h.shape[c]]),f=et({inputs:{x:h},backend:e,attrs:{shape:[-1,p]}});l.push(f);const m=Qh(o.dtype),g=(y,$,v,N,T)=>{const k=y.shape[0],S=y.shape[1],C=tb(S,T),D={windowSize:C,inSize:S,batchSize:k,numSegments:T},O=new lK(D,$),P=e.compileAndRun(O,[y,v],N);if(l.push(P),P.shape[1]===T)return P;const M=i$({backend:e,attrs:{start:0,stop:T,step:1,dtype:"float32"}}),V=a$({inputs:{x:M},backend:e,attrs:{reps:[S/C]}});return l.push(M),l.push(V),g(P,$,V,N,T)},x=g(f,"unsortedSegmentSum",r,m,i),b=et({inputs:{x},backend:e,attrs:{shape:d}});let w=b;if(u!=null){l.push(b);const y=Ps(u);w=_e({inputs:{x:w},backend:e,attrs:{perm:y}})}return l.forEach(y=>e.disposeIntermediateTensorInfo(y)),w}const uK={kernelName:mc,backendName:"webgl",kernelFunc:cK};const hK=[KB,YB,QB,nV,oV,aV,cV,hV,mV,xV,wV,IV,SV,RV,FV,OV,MV,VV,UV,HV,XV,nW,oW,lW,uW,gW,bW,CW,AB,kW,RW,_W,BW,UW,HW,jW,XW,QW,n4,r4,a4,c4,h4,f4,g4,w4,C4,k4,T4,R4,_4,P4,W4,H4,K4,X4,Z4,Q4,eU,sU,rU,cU,dU,mU,xU,wU,IU,NU,AU,RB,FU,TW,LU,zU,WU,FB,qU,YU,JU,nG,rG,cG,dG,gG,wG,IG,kG,EG,AG,FG,MG,zG,VG,UG,HG,XG,QG,sH,hH,LB,mH,bH,$H,vH,dW,NH,EH,AH,_H,PH,OB,BH,WH,GH,qH,jH,pW,aH,YH,tq,oq,PB,lq,hq,mq,bq,Cq,vq,Nq,Rq,Fq,Lq,zq,Wq,qq,Xq,Qq,nj,tW,cH,rj,aj,cj,hj,pj,mj,xj,yj,$j,vj,Sj,Tj,Rj,Fj,Oj,Mj,zj,lH,HB,Wj,Hj,jj,Yj,tK,sK,qB,rK,aK,uK,TH];for(const n of hK)Zh(n);export{ct as $,sh as A,ch as B,Fn as C,eg as D,R as E,ZK as F,Ct as G,rS as H,Ki as I,go as J,ye as K,ng as L,sg as M,Ju as N,yc as O,ix as P,cg as Q,rg as R,ug as S,Lt as T,cx as U,Oe as V,Xi as W,dd as X,Q$ as Y,mt as Z,rt as _,I as a,gp as a$,fo as a0,lg as a1,hg as a2,we as a3,Bo as a4,_d as a5,ud as a6,Fd as a7,Te as a8,Ge as a9,gC as aA,CC as aB,yf as aC,Ix as aD,Ff as aE,VK as aF,VN as aG,GK as aH,BK as aI,zK as aJ,WK as aK,WN as aL,Xg as aM,st as aN,vt as aO,R$ as aP,af as aQ,D$ as aR,Od as aS,Ld as aT,Md as aU,Ib as aV,Sb as aW,kb as aX,vb as aY,gb as aZ,xp as a_,Ce as aa,Qt as ab,gt as ac,mo as ad,dg as ae,co as af,Ad as ag,r2 as ah,s2 as ai,Dd as aj,e2 as ak,Ef as al,xC as am,PK as an,Zt as ao,B as ap,ls as aq,mC as ar,od as as,qN as at,fC as au,UK as av,Ng as aw,MK as ax,gf as ay,IC as az,Tt as b,D0 as b$,bp as b0,yp as b1,wb as b2,wp as b3,$p as b4,mp as b5,$b as b6,bb as b7,xb as b8,yb as b9,N0 as bA,b0 as bB,V0 as bC,Fp as bD,l0 as bE,_0 as bF,T0 as bG,U0 as bH,W0 as bI,ry as bJ,ly as bK,iy as bL,cy as bM,C0 as bN,Rp as bO,oa as bP,q0 as bQ,i0 as bR,I0 as bS,Bc as bT,F0 as bU,Y0 as bV,Q0 as bW,ny as bX,P0 as bY,z0 as bZ,L0 as b_,Xe as ba,mb as bb,BR as bc,Uo as bd,hn as be,Vr as bf,E0 as bg,O0 as bh,G0 as bi,M0 as bj,Z0 as bk,ty as bl,sy as bm,H0 as bn,dy as bo,my as bp,fy as bq,B0 as br,Pc as bs,ia as bt,d0 as bu,aa as bv,p0 as bw,k0 as bx,Dp as by,g0 as bz,E as c,Er as c$,a0 as c0,yy as c1,r0 as c2,R0 as c3,py as c4,A0 as c5,gy as c6,Ws as c7,m0 as c8,$0 as c9,Pr as cA,yt as cB,qs as cC,pn as cD,eo as cE,hm as cF,rC as cG,he as cH,KC as cI,YC as cJ,Mg as cK,Su as cL,Fr as cM,eI as cN,sI as cO,rI as cP,aI as cQ,cI as cR,hI as cS,id as cT,yI as cU,$c as cV,EI as cW,AI as cX,FI as cY,ad as cZ,OI as c_,Ep as ca,u0 as cb,S0 as cc,Ap as cd,c0 as ce,hy as cf,x0 as cg,j0 as ch,It as ci,zc as cj,kp as ck,_c as cl,Ab as cm,cA as cn,Db as co,_b as cp,Rb as cq,vp as cr,lA as cs,l6 as ct,a6 as cu,uA as cv,c6 as cw,p6 as cx,Lc as cy,f6 as cz,ke as d,Nn as d$,PI as d0,on as d1,ro as d2,VI as d3,UI as d4,HI as d5,jI as d6,Pg as d7,ho as d8,zg as d9,vc as dA,Lg as dB,QK as dC,Ze as dD,So as dE,Xa as dF,pd as dG,ts as dH,ox as dI,Gv as dJ,qv as dK,Kv as dL,fd as dM,qa as dN,nr as dO,ON as dP,Qv as dQ,mn as dR,md as dS,ik as dT,jg as dU,gd as dV,Gn as dW,xd as dX,Kg as dY,fk as dZ,n6 as d_,QI as da,nv as db,cd as dc,Vg as dd,ux as de,Tu as df,Wg as dg,wf as dh,cv as di,dv as dj,xv as dk,yv as dl,ZS as dm,xr as dn,Cc as dp,JS as dq,fn as dr,Ug as ds,Av as dt,rn as du,Me as dv,Ov as dw,qg as dx,Nd as dy,qi as dz,J as e,Bn as e$,bd as e0,xk as e1,zs as e2,te as e3,Ha as e4,jo as e5,Ck as e6,vk as e7,yd as e8,Ht as e9,Sd as eA,ji as eB,$d as eC,s6 as eD,t6 as eE,sn as eF,Ne as eG,Vt as eH,rx as eI,Hn as eJ,Yi as eK,LS as eL,o6 as eM,it as eN,PS as eO,Be as eP,Fa as eQ,kn as eR,BS as eS,ax as eT,US as eU,lx as eV,xo as eW,qS as eX,Se as eY,pe as eZ,kt as e_,Ic as ea,Lr as eb,gn as ec,_k as ed,Cd as ee,BC as ef,Mk as eg,YK as eh,Mr as ei,ja as ej,cS as ek,ds as el,Yg as em,Td as en,Zg as eo,Jg as ep,Qg as eq,tx as er,bS as es,e6 as et,ex as eu,nx as ev,vd as ew,sx as ex,kd as ey,Ka as ez,A as f,mu as f$,z as f0,Kr as f1,xt as f2,_p as f3,We as f4,CD as f5,W$ as f6,uD as f7,un as f8,b6 as f9,E$ as fA,KR as fB,ni as fC,si as fD,oi as fE,uh as fF,Il as fG,vl as fH,kl as fI,hh as fJ,Sl as fK,Nl as fL,dh as fM,ph as fN,ri as fO,ii as fP,mh as fQ,fh as fR,Tl as fS,ZR as fT,Gm as fU,gh as fV,xh as fW,El as fX,bh as fY,yh as fZ,Rl as f_,y6 as fa,Mb as fb,gl as fc,Xr as fd,Yr as fe,mx as ff,gx as fg,xx as fh,bx as fi,tr as fj,oh as fk,rh as fl,xl as fm,bl as fn,Zr as fo,Jr as fp,Qr as fq,ei as fr,ti as fs,yl as ft,wl as fu,ah as fv,ih as fw,$l as fx,Cl as fy,lh as fz,er as g,Ah as g$,fu as g0,Jm as g1,wh as g2,li as g3,$h as g4,I$ as g5,Al as g6,ci as g7,ui as g8,Dl as g9,Ml as gA,Pl as gB,yi as gC,wi as gD,A$ as gE,zl as gF,Bl as gG,Vl as gH,pK as gI,fK as gJ,Uc as gK,qc as gL,mK as gM,Ul as gN,Gl as gO,Hl as gP,Rh as gQ,Eh as gR,$i as gS,ql as gT,jl as gU,Ci as gV,Kl as gW,Ii as gX,yx as gY,vi as gZ,Xl as g_,hi as ga,Ch as gb,Ih as gc,vh as gd,di as ge,pi as gf,Fl as gg,Ua as gh,hu as gi,kh as gj,_l as gk,Ol as gl,fi as gm,YR as gn,Sh as go,mi as gp,Nh as gq,ae as gr,gi as gs,xi as gt,bi as gu,Yu as gv,Wl as gw,Th as gx,DR as gy,Ll as gz,tg as h,Kh as h$,Dh as h0,Fh as h1,Yl as h2,Jl as h3,Zl as h4,Vs as h5,JN as h6,Ql as h7,tc as h8,gK as h9,Di as hA,Fi as hB,lc as hC,dc as hD,Li as hE,uc as hF,Ph as hG,zh as hH,Bh as hI,Vh as hJ,hc as hK,Mi as hL,Wh as hM,Pi as hN,pc as hO,Ui as hP,Uh as hQ,Gh as hR,Hh as hS,qh as hT,zi as hU,cc as hV,qn as hW,Bi as hX,Vi as hY,Wi as hZ,jh as h_,ki as ha,ec as hb,nc as hc,wx as hd,_h as he,df as hf,Oh as hg,ai as hh,Si as hi,ze as hj,Ni as hk,Ti as hl,sc as hm,rc as hn,Mh as ho,oc as hp,Lh as hq,ic as hr,Yh as hs,Ei as ht,Ri as hu,zd as hv,ac as hw,Ai as hx,Oi as hy,_i as hz,w$ as i,zo as i0,Xh as i1,fc as i2,mc as i3,xK as i4,Ga as i5,gc as i6,Wa as i7,Sg as i8,sT as i9,kg as iA,_$ as iB,Zh as iC,FK as iD,JK as iE,r6 as iF,AK as iG,LK as iH,s3 as iI,BF as iJ,ZN as iK,Qh as iL,CK as iM,RK as iN,_o as iO,yK as iP,bK as iQ,Ke as iR,$K as iS,KK as iT,XK as iU,nk as iV,C6 as iW,w6 as iX,HK as ia,wK as ib,NK as ic,IK as id,SK as ie,TK as ig,kK as ih,vK as ii,On as ij,_K as ik,OK as il,EB as im,i6 as io,lC as ip,lf as iq,gu as ir,$6 as is,qK as it,jK as iu,u6 as iv,mf as iw,Dx as ix,EK as iy,DK as iz,U as j,hs as k,nd as l,Et as m,og as n,_ as o,wd as p,ig as q,L as r,_t as s,wc as t,ag as u,es as v,Qo as w,je as x,bc as y,Es as z};
