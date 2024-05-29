// import { loadPyodideAndPackages } from './pyodide_worker.mjs';
import { expose, wrap, proxy } from 'comlink';
import { loadPyodide, version } from 'pyodide';
import type { PyodideInterface } from 'pyodide';
import type { Server as FitServer } from './standalone_fit_worker';

const DEBUG = true;

var pyodide: PyodideInterface;

declare const REFL1D_WHEEL_FILE: string;
declare const BUMPS_WHEEL_FILE: string;

async function loadPyodideAndPackages() { // loads pyodide
    pyodide = await loadPyodide({
        indexURL: `https://cdn.jsdelivr.net/pyodide/v${version}/full/`
    }); // run the function and wait for the result (base library)

    await pyodide.loadPackage(["numpy", "scipy", "pytz", "h5py", "micropip"]); // waits until these python packpages are loaded to continue
  
    //import reductus library with micropip
    let api = await pyodide.runPythonAsync(`
    import micropip
    await micropip.install([
        "matplotlib",
        "plotly",
        "mpld3",
        "periodictable",
        "blinker",
        "dill",
    ])
    await micropip.install("./wheels/${BUMPS_WHEEL_FILE}")
    await micropip.install("./wheels/${REFL1D_WHEEL_FILE}", keep_going=True, deps=False)

    print("pip imports finished")
    from bumps.webview.server import api
    from refl1d.webview.server import api as refl1d_api
    api.state.parallel = 0
    api.state.problem.serializer = "dataclass"
    print("api imported")
    import refl1d
    import json
    import dill
    # setup backend:
    refl1d.use('c_ext')
    await api.emit("add_notification", {
        "title": "Backend Ready",
        "content": f"All packages loaded",
        "timeout": 2000,
    })

    wrapped_api = {}
    fit_server = object()

    def expose(method, method_name):
        def wrapper(args):
            # print("args:", args)
            pyargs = args.to_py() if args is not None else []
            #print(method_name, "pyargs:", pyargs)
            result = method(*pyargs)
            print("result of", method_name, str(result))
            return result

        return wrapper

    for method_name, method in api.REGISTRY.items():
        if method_name == "start_fit_thread":
            continue
        print("wrapping:", method_name)
        wrapped = expose(method, method_name)
        wrapped_api[method_name] = wrapped

    wrapped_progress = expose(api._fit_progress_handler, "evt_fit_progress")
    wrapped_complete = expose(api._fit_complete_handler, "evt_fit_complete")
    wrapped_api["evt_fit_progress"] = wrapped_progress
    wrapped_api["evt_fit_complete"] = wrapped_complete

    async def start_fit_thread(fitter_id: str="", options=None, terminate_on_finish=False):
        print("starting fit thread", fitter_id, options)
        state = api.state
        fitProblem = state.problem.fitProblem if state.problem is not None else None
        if fitProblem is None:
            await api.log("Error: Can't start fit if no problem loaded")
        else:
            if state.fit_thread is not None:
                # warn that fit is alread running...
                logger.warning("fit already running...")
                await log("Can't start fit, a fit is already running...")
                return
            # TODO: better access to model parameters
            num_params = len(fitProblem.getp())
            if num_params == 0:
                raise ValueError("Problem has no fittable parameters")

            dumped = dill.dumps(fitProblem)
            await api.emit("set_fit_thread_problem", dumped)
            num_steps = api.get_num_steps(fitter_id, num_params, options)
            state.fit_stored_problem = fitProblem
            fit_thread = fitter_id

            await api.emit("fit_progress", {}) # clear progress
            state.shared.active_fit = api.to_json_compatible_dict(dict(fitter_id=fitter_id, options=options, num_steps=num_steps))
            await api.log(json.dumps(api.to_json_compatible_dict(options), indent=2), title = f"starting fitter {fitter_id}")
            state.autosave()
            await api.emit("add_notification", {
                "title": "Fit Started",
                "content": f"Fit started with problem: {api.state.problem.fitProblem.name}",
                "timeout": 2000,
            });
            await api.emit("start_fit_thread_fit", fitter_id, options, terminate_on_finish)

    wrapped_api["start_fit_thread"] = expose(start_fit_thread, "start_fit_thread")

    wrapped_api
    `);
    return api;
}

// export { loadPyodideAndPackages };

let pyodideReadyPromise = loadPyodideAndPackages(); // run the functions stored in lines 4
const fit_worker = new Worker(new URL("./standalone_fit_worker.ts", import.meta.url), {type: 'module'});
const FitServerClass = wrap<FitServer>(fit_worker);
const FitServerPromise = new FitServerClass();
const FitSignals = ["progress"]
type EventCallback = (message?: any) => any;

export class Server {
    handlers: { [signal: string]: EventCallback[] }
    nativefs: any;

    constructor() {
        this.handlers = {};
        this.nativefs = null;
        this.init();
    }

    async init() {
        const api = await pyodideReadyPromise;
        const fit_server = await FitServerPromise;
        const defineEmit = await pyodide.runPythonAsync(`
            def defineEmit(server):
                api.emit = server.asyncEmit;
            
            defineEmit
         `);
        await defineEmit(this);
        const define_fit_server = await pyodide.runPythonAsync(`
            def define_fit_server(fit_server_js):
                global fit_server
                fit_server = fit_server_js
                            
            define_fit_server
        `);
        await define_fit_server(fit_server);
        this.addHandler('set_fit_thread_problem', async (problem: any) => {
            console.log('set_fit_thread_problem:', problem);
            console.log({fit_server}, fit_server.onAsyncEmit);
            const result = await fit_server.onAsyncEmit('set_problem', problem);
            console.log("result:", result);
        });
        this.addHandler('start_fit_thread_fit', async (...args: any[]) => {
            console.log('start_fit_thread_fit:', args);
            const result = await fit_server.onAsyncEmit('start_fit_thread', ...args);
            console.log("result:", result);
        });
        const fit_progress_handler = async (event: any) => {
            console.log('evt_fit_progress:', event);
            await this.onAsyncEmit('evt_fit_progress', event);
        }
        fit_server.addHandler('evt_fit_progress', proxy(fit_progress_handler));
        const fit_complete_handler = async (event: any) => {
            console.log('evt_fit_complete:', event);
            await this.onAsyncEmit('evt_fit_complete', event);
        }
        fit_server.addHandler('evt_fit_complete', proxy(fit_complete_handler));
    }

    async addHandler(signal: string, handler: EventCallback) {
        const signal_handlers = this.handlers[signal] ?? [];
        signal_handlers.push(handler);
        if (DEBUG) {
            console.log(`adding handler: ${signal}`);
        }
        if (signal === 'connect') {
            await pyodideReadyPromise;
            await FitServerPromise;
            await handler();
        }
        this.handlers[signal] = signal_handlers;
    }

    async removeHandler(signal: string, handler: EventCallback) {
        let signal_handlers = this.handlers[signal] ?? [];
        signal_handlers = signal_handlers.filter((h) => {
            if (h === handler) {
                console.log('matching worker handler found, removing: ', handler);
                return false;
            }
            return true;
        })
        this.handlers[signal] = signal_handlers;
    }

    async mount(dirHandle: FileSystemDirectoryHandle) {
        // const dirHandle = await self.showDirectoryPicker();
        console.log({dirHandle});   
        const nativefs = await pyodide.mountNativeFS("/home/pyodide/user_mount", dirHandle);
        this.nativefs = nativefs;
        await pyodide.runPythonAsync(`
        import os
        os.chdir("/home/pyodide/user_mount")

        `);
    }

    async syncFS() {
        let r = await this.nativefs?.syncfs?.();
    }

    async asyncEmit(signal: string, ...args: unknown[]) {
        // this is for emit() calls from the python server
        const js_args = args.map((arg) => {
            return arg?.toJs?.({dict_converter: Object.fromEntries}) ?? arg;
        });
        // const jsMessage = message?.toJs?.({dict_converter: Object.fromEntries}) ?? message;
        console.log('server emit:', signal, js_args);
        const handlers = this.handlers[signal] ?? [];
        for (let handler of handlers) {
            handler(...js_args);
        }
    }

    async onAsyncEmit(signal: string, ...args: any[]) {
        // this is for emit() calls from the client
        const api = await pyodideReadyPromise;
        const callback = (args[args.length - 1] instanceof Function) ? args.pop() : null;
        const result = await api.get(signal)(args);
        const jsResult = result?.toJs?.({dict_converter: Object.fromEntries}) ?? result;
        if (callback !== null) {
            await callback(jsResult);
        }
        return jsResult;
    }

}

expose(Server);
