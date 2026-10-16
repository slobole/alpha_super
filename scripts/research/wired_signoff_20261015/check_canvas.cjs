// Check the saved canvas against the installed SDK without running its UI.
const fs_module = require('node:fs');
const path_module = require('node:path');
const crypto_module = require('node:crypto');
const ts_module = require('C:/Users/User/AppData/Local/Programs/cursor/resources/app/extensions/node_modules/typescript/lib/typescript.js');
const repo_path_str = process.cwd();
const output_path_str = path_module.join(repo_path_str, 'results/review/wired_signoff_20261015');
const canvas_path_str = path_module.join(output_path_str, 'wired-signoff.canvas.tsx');
const sdk_path_str = 'C:/Users/User/.cursor/skills-cursor/canvas/sdk/index.d.ts';
const shim_path_str = path_module.join(output_path_str, 'canvas-check-shim.d.ts');
// Only React's rendering types are shimmed; actual canvas exports/props come
// from the installed SDK. This is an offline type check, not runtime rendering.
const shim_str = `declare namespace JSX { interface Element {} interface ElementChildrenAttribute { children: {}; } interface IntrinsicAttributes { key?: string|number; } interface IntrinsicElements { [name: string]: any; } }
declare module "react" { export type ReactNode=any; export type CSSProperties={ [key:string]:string|number|undefined }; export type RefObject<T>=any; export type ComponentType<T>=any; export function useState<T>(initial:T):[T,(value:T)=>void]; export function useMemo<T>(fn:()=>T,deps:any[]):T; export function useEffect(fn:()=>void,deps:any[]):void; export function useRef<T>(value:T):any; }
`;
const compiler_option_dict = {target:ts_module.ScriptTarget.ES2022,module:ts_module.ModuleKind.ESNext,
  moduleResolution:ts_module.ModuleResolutionKind.Bundler,jsx:ts_module.JsxEmit.Preserve,
  strict:true,skipLibCheck:true,noEmit:true,paths:{'cursor/canvas':[sdk_path_str]}};
const host_obj = ts_module.createCompilerHost(compiler_option_dict);
const original_read_fn = host_obj.readFile;
const original_exists_fn = host_obj.fileExists;
const is_shim_fn = file_path_str => path_module.normalize(file_path_str).toLowerCase() === path_module.normalize(shim_path_str).toLowerCase();
host_obj.readFile = file_path_str => is_shim_fn(file_path_str) ? shim_str : original_read_fn(file_path_str);
host_obj.fileExists = file_path_str => is_shim_fn(file_path_str) || original_exists_fn(file_path_str);
const program_obj = ts_module.createProgram([canvas_path_str,shim_path_str],compiler_option_dict,host_obj);
const diagnostic_list = ts_module.getPreEmitDiagnostics(program_obj).map(item_obj=>({
  code:item_obj.code,file:item_obj.file?.fileName,
  line:item_obj.file && item_obj.start != null ? item_obj.file.getLineAndCharacterOfPosition(item_obj.start).line+1:null,
  message:ts_module.flattenDiagnosticMessageText(item_obj.messageText,'\n')}));
const payload_dict = {status:diagnostic_list.length?'error':'pass',diagnostics:diagnostic_list,
  canvas_sha256:crypto_module.createHash('sha256').update(fs_module.readFileSync(canvas_path_str)).digest('hex'),
  sdk_path:sdk_path_str,compiler_version:ts_module.version,
  limitation:'Uses installed canvas SDK exports/props and a small React type shim. No runtime canvas renderer was invoked.'};
fs_module.writeFileSync(path_module.join(output_path_str,'canvas_typecheck.json'),JSON.stringify(payload_dict,null,2));
process.stdout.write(JSON.stringify(payload_dict));
process.exitCode=diagnostic_list.length?1:0;
