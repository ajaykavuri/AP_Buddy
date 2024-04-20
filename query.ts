import { main as main } from './setup'
import { agent as agent } from './setup'
async function query() {
    const newAgent = await agent;
    main("What is the formula for a solid of revolution for AP Calculus BC?", newAgent);
    return;
}

query();