import { app } from "../../scripts/app.js";

const NODE_CLASS = "API_Group_Muter";
const EMPTY_GROUP_VALUE = "No groups found";
const MODE_ALWAYS = 0;
const MODE_NEVER = 2;
const STATE_KEY = "__api_group_muter_state";

function getGraph() {
    return app.canvas?.getCurrentGraph?.() || app.canvas?.graph || app.graph;
}

function getRootGraph() {
    const graph = getGraph();
    return app.rootGraph || graph?.rootGraph || app.graph?.rootGraph || app.graph || graph;
}

function getGraphs(graph = getGraph()) {
    const seen = new Set();
    const graphs = [];
    const rootGraph = getRootGraph();

    for (const candidate of [graph, rootGraph]) {
        if (candidate && !seen.has(candidate)) {
            seen.add(candidate);
            graphs.push(candidate);
        }
    }

    for (const candidateGraph of [...graphs]) {
        const subgraphs = candidateGraph?.subgraphs?.values?.();
        if (!subgraphs) {
            continue;
        }
        for (const subgraph of subgraphs) {
            if (subgraph && !seen.has(subgraph)) {
                seen.add(subgraph);
                graphs.push(subgraph);
            }
        }
    }

    return graphs;
}

function asArray(value) {
    if (!value) {
        return [];
    }
    if (Array.isArray(value)) {
        return value;
    }
    if (typeof value.values === "function") {
        return Array.from(value.values());
    }
    if (typeof value.length === "number") {
        return Array.from(value);
    }
    if (typeof value === "object") {
        return Object.values(value);
    }
    return [];
}

function getGroups(graph = getGraph()) {
    const seen = new Set();
    const groups = [];

    for (const candidateGraph of getGraphs(graph)) {
        for (const group of asArray(candidateGraph?._groups ?? candidateGraph?.groups)) {
            if (!seen.has(group)) {
                seen.add(group);
                groups.push(group);
            }
        }
    }

    if (!groups.length) {
        groups.push(...asArray(graph?.serialize?.()?.groups));
    }

    return groups;
}

function getNodes(graph = getGraph()) {
    return asArray(graph?._nodes ?? graph?.nodes);
}

function getAllNodes(graph = getGraph()) {
    const seen = new Set();
    const nodes = [];

    for (const candidateGraph of getGraphs(graph)) {
        for (const node of getNodes(candidateGraph)) {
            if (!seen.has(node)) {
                seen.add(node);
                nodes.push(node);
            }
        }
    }

    return nodes;
}

function getNodeById(graph, id) {
    const found = (
        graph?.getNodeById?.(id) ||
        graph?.getNodeById?.(Number(id)) ||
        graph?._nodes_by_id?.[id] ||
        graph?._nodes_by_id?.get?.(id) ||
        graph?._nodes_by_id?.get?.(Number(id))
    );
    if (found) {
        return found;
    }

    for (const candidateGraph of getGraphs(graph)) {
        const node =
            candidateGraph?.getNodeById?.(id) ||
            candidateGraph?.getNodeById?.(Number(id)) ||
            candidateGraph?._nodes_by_id?.[id] ||
            candidateGraph?._nodes_by_id?.get?.(id) ||
            candidateGraph?._nodes_by_id?.get?.(Number(id));
        if (node) {
            return node;
        }
    }
}

function getGroupTitle(group, index) {
    return String(group?.title || group?.name || `Group ${index + 1}`);
}

function getGroupEntries(graph = getGraph()) {
    const groups = getGroups(graph);
    const totals = {};
    const seen = {};

    for (let i = 0; i < groups.length; i++) {
        const title = getGroupTitle(groups[i], i);
        totals[title] = (totals[title] || 0) + 1;
    }

    return groups.map((group, index) => {
        const title = getGroupTitle(group, index);
        seen[title] = (seen[title] || 0) + 1;
        return {
            group,
            title,
            value: totals[title] > 1 ? `${title} (${seen[title]})` : title,
        };
    });
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getGroupValues(graph = getGraph()) {
    const entries = getGroupEntries(graph);
    return entries.length ? entries.map((entry) => entry.value) : [EMPTY_GROUP_VALUE];
}

function updateComboWidget(node) {
    const groupWidget = getWidget(node, "group");
    if (!groupWidget) {
        return;
    }

    const graph = node.graph || getGraph();
    const values = getGroupValues(graph);
    groupWidget.options ??= {};
    groupWidget.options.values = () => getGroupValues(node.graph || getGraph());

    if (!values.includes(groupWidget.value)) {
        groupWidget.value = values[0];
    }
}

function getRect(item) {
    if (!item) {
        return null;
    }

    const bounds = item.boundingRect || item.bounding || item._bounding || item.rect;
    if (bounds?.length >= 4) {
        return [Number(bounds[0]), Number(bounds[1]), Number(bounds[2]), Number(bounds[3])];
    }

    const pos = item.pos || item._pos || [item.x, item.y];
    const size = item.size || item._size || [item.width, item.height];
    if (!pos || !size) {
        return null;
    }

    return [Number(pos[0]), Number(pos[1]), Number(size[0]), Number(size[1])];
}

function rectsOverlap(a, b) {
    if (!a || !b || a.some(Number.isNaN) || b.some(Number.isNaN)) {
        return false;
    }

    return (
        a[0] < b[0] + b[2] &&
        a[0] + a[2] > b[0] &&
        a[1] < b[1] + b[3] &&
        a[1] + a[3] > b[1]
    );
}

function rectContainsPoint(rect, point) {
    if (!rect || !point || rect.some(Number.isNaN) || point.some(Number.isNaN)) {
        return false;
    }

    return (
        point[0] >= rect[0] &&
        point[0] < rect[0] + rect[2] &&
        point[1] >= rect[1] &&
        point[1] < rect[1] + rect[3]
    );
}

function findGroup(value, graph = getGraph()) {
    return getGroupEntries(graph).find((entry) => entry.value === value)?.group || null;
}

function getNodeBounding(node) {
    let bounds = node?.getBounding?.();
    if (
        bounds &&
        bounds[0] === 0 &&
        bounds[1] === 0 &&
        bounds[2] === 0 &&
        bounds[3] === 0
    ) {
        const ctx = node.graph?.primaryCanvas?.canvas?.getContext?.("2d") || app.canvas?.canvas?.getContext?.("2d");
        if (ctx) {
            node.updateArea?.(ctx);
            bounds = node.getBounding?.();
        }
    }

    return getRect({ boundingRect: bounds }) || getRect(node);
}

function recomputeGroupNodes(group, graph = getGraph()) {
    const groupGraph = group?.graph || graph;
    const groupRect = getRect(group);
    if (!groupRect || !groupGraph) {
        return;
    }

    group._children?.clear?.();
    if (Array.isArray(group.nodes)) {
        group.nodes.length = 0;
    }

    for (const node of getNodes(groupGraph)) {
        const nodeRect = getNodeBounding(node);
        const center = nodeRect && [nodeRect[0] + nodeRect[2] * 0.5, nodeRect[1] + nodeRect[3] * 0.5];
        if (center && rectContainsPoint(groupRect, center)) {
            group._children?.add?.(node);
            if (Array.isArray(group.nodes)) {
                group.nodes.push(node);
            }
        }
    }
}

function getGroupNodes(group, controllerNode, graph = getGraph()) {
    recomputeGroupNodes(group, graph);

    const children = asArray(group?._children);
    if (children.length) {
        return children.filter((node) => node && node.id !== controllerNode.id);
    }

    const allNodes = getNodes(group?.graph || graph);
    const explicitNodes = group?.nodes || group?._nodes;

    if (Array.isArray(explicitNodes) && explicitNodes.length) {
        return explicitNodes
            .map((entry) => {
                if (entry && typeof entry === "object") {
                    return entry;
                }
                return getNodeById(graph, entry);
            })
            .filter((node) => node && node.id !== controllerNode.id);
    }

    const groupRect = getRect(group);
    return allNodes.filter((node) => {
        return node?.id !== controllerNode.id && rectsOverlap(getRect(node), groupRect);
    });
}

function setNodeMode(node, mode) {
    if (!node || node.mode === mode) {
        return;
    }

    node.mode = mode;
    node.graph?.incrementVersion?.();
}

function getState(node) {
    node.properties ??= {};
    node.properties[STATE_KEY] ??= { group: null, modes: {} };
    node.properties[STATE_KEY].modes ??= {};
    return node.properties[STATE_KEY];
}

function restoreTrackedNodes(node, keepIds = new Set()) {
    const graph = node.graph || getGraph();
    const state = getState(node);

    for (const [nodeId, mode] of Object.entries(state.modes)) {
        if (keepIds.has(String(nodeId))) {
            continue;
        }

        const targetNode = getNodeById(graph, nodeId);
        if (targetNode) {
            setNodeMode(targetNode, Number(mode));
        }
        delete state.modes[nodeId];
    }
}

function applyGroupMute(node) {
    updateComboWidget(node);

    const groupWidget = getWidget(node, "group");
    const muteWidget = getWidget(node, "mute");
    const state = getState(node);
    const selectedGroup = groupWidget?.value;
    const enabled = Boolean(muteWidget?.value);
    const graph = node.graph || getGraph();
    const group = enabled && selectedGroup !== EMPTY_GROUP_VALUE ? findGroup(selectedGroup, graph) : null;

    if (!group) {
        restoreTrackedNodes(node);
        state.group = null;
        redraw();
        return;
    }

    const groupNodes = getGroupNodes(group, node, graph);
    const activeIds = new Set(groupNodes.map((targetNode) => String(targetNode.id)));

    restoreTrackedNodes(node, activeIds);

    for (const targetNode of groupNodes) {
        const nodeId = String(targetNode.id);
        if (!(nodeId in state.modes)) {
            state.modes[nodeId] = targetNode.mode ?? MODE_ALWAYS;
        }
        setNodeMode(targetNode, MODE_NEVER);
    }

    state.group = selectedGroup;
    redraw();
}

function redraw() {
    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function scheduleApply(node) {
    clearTimeout(node.__apiGroupMuterTimer);
    node.__apiGroupMuterTimer = setTimeout(() => applyGroupMute(node), 0);
}

function installWidgetCallbacks(node) {
    for (const widgetName of ["group", "mute"]) {
        const widget = getWidget(node, widgetName);
        if (!widget || widget.__apiGroupMuterInstalled) {
            continue;
        }

        const originalCallback = widget.callback;
        widget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            scheduleApply(node);
            return result;
        };
        widget.__apiGroupMuterInstalled = true;
    }
}

function installRemovalHandler(node) {
    if (node.__apiGroupMuterRemovalInstalled) {
        return;
    }

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        restoreTrackedNodes(node);
        return originalOnRemoved?.apply(this, arguments);
    };
    node.__apiGroupMuterRemovalInstalled = true;
}

function refreshAllControllers() {
    for (const node of getAllNodes()) {
        if (node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS) {
            installWidgetCallbacks(node);
            installRemovalHandler(node);
            scheduleApply(node);
        }
    }
}

app.registerExtension({
    name: "api.group_muter",

    async nodeCreated(node) {
        if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) {
            return;
        }

        installWidgetCallbacks(node);
        installRemovalHandler(node);
        scheduleApply(node);
    },

    async afterConfigureGraph() {
        refreshAllControllers();
    },

    async loadedGraphNode(node) {
        if (node?.comfyClass !== NODE_CLASS && node?.type !== NODE_CLASS) {
            return;
        }

        installWidgetCallbacks(node);
        installRemovalHandler(node);
        scheduleApply(node);
    },

    async setup() {
        setInterval(refreshAllControllers, 1000);
    },
});
