import { wrap, proxy } from 'comlink';
import type { Endpoint, Remote } from 'comlink';
// import './standalone_worker';
import type { Server } from './standalone_worker';
// import type { Server as FitServer } from './standalone_fit_worker';

export function io() {
  const worker = new Worker(new URL("./standalone_worker.ts", import.meta.url), {type: 'module'});
  // const fit_worker = new Worker(new URL("./standalone_fit_worker.ts", import.meta.url), {type: 'module'});
  const socket = new AsyncSocket(worker);
  return socket;
};

type EventCallback = (message: any) => any;

export class AsyncSocket {
  proxy_lookups: Map<EventCallback, typeof proxy<EventCallback>>;
  handlers: { [signal: string]: EventCallback[] }
  ServerPromise: Promise<Remote<Server>>
  // FitServerPromise: Promise<Remote<FitServer>>
  worker: Worker
  // fit_worker: Worker
  id: string = "WebWorker"

  constructor(worker: Worker) {
    this.handlers = {};
    this.proxy_lookups = new Map();
    this.worker = worker;
    // this.fit_worker = fit_worker;
    const ServerClass = wrap<Server>(worker);
    // const FitServerClass = wrap<FitServer>(fit_worker);
    this.ServerPromise = new ServerClass();
    // this.FitServerPromise = new FitServerClass();
  }
  connect() {}
  on(signal: string, handler: EventCallback) {
    const proxy_handler = proxy(handler);
    this.proxy_lookups.set(handler, proxy_handler);
    return this.ServerPromise.then((server) => {
      return server.addHandler(signal, proxy(handler));
    });
  }
  off(signal: string, handler: EventCallback) {
    const proxy_handler = this.proxy_lookups.get(handler);
    if (proxy_handler !== undefined) {
      console.log("removing handler: ", handler);
      return this.ServerPromise.then((server) => {
        return server.removeHandler(signal, proxy_handler);
      })
    }
  }
  async mountLocal() {
    const dirHandle = await window.showDirectoryPicker({mode: "readwrite"});
    console.log({dirHandle});
    
    const server = await this.ServerPromise;
    await server.mount(dirHandle);
    await server.asyncEmit('add_notification', {content: dirHandle.name, title: "Local Directory Mounted", timeout: 3000 });
    return true;
  }
  async syncFS() {
    const server = await this.ServerPromise;
    return await server.syncFS();
  }
  async asyncEmit(signal: string, ...args: any[]) {
    const server = await this.ServerPromise;
    console.log({server, signal, args});
    let last_arg = args.pop();
    if (last_arg instanceof Function) {
      last_arg = proxy(last_arg);
    }
    if (last_arg !== undefined) {
      args.push(last_arg);
    }
    return await server.onAsyncEmit(signal, ...args);
  }

}

export const Socket = AsyncSocket;