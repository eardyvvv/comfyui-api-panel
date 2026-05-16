import { app } from "../../scripts/app.js";

const NODE_CLASS = "API_Group_Muter";
const EMPTY_GROUP_VALUE = "No groups found";
const MODE_ALWAYS = 0;
const MODE_NEVER = 2;
const STATE_KEY = "__api_group_muter_state";

function getGraph() {
    return app.canvas?.graph || app.graph;
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
    const liveGroups = asArray(graph?._groups ?? graph?.groups);
    if (liveGroups.length) {
        return liveGroups;
    }

    return asArray(graph?.serialize?.()?.groups);
}

function getNodes(graph = getGraph()) {
    return asArray(graph?._nodes ?? graph?.nodes);
}

function getNodeById(graph, id) {
    return (
        graph?.getNodeById?.(id) ||
        graph?.getNodeById?.(Number(id)) ||
        graph?._nodes_by_id?.[id] ||
        graph?._nodes_by_id?.get?.(id) ||
        graph?._nodes_by_id?.get?.(Number(id))
    );
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

function updateComboWidget(node) {
    const groupWidget = getWidget(node, "group");
    if (!groupWidget) {
        return;
    }

    const entries = getGroupEntries(node.graph || getGraph());
    const values = entries.length ? entries.map((entry) => entry.value) : [EMPTY_GROUP_VALUE];
    groupWidget.options ??= {};
    groupWidget.options.values = values;

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

function findGroup(value, graph = getGraph()) {
    return getGroupEntries(graph).find((entry) => entry.value === value)?.group || null;
}

function getGroupNodes(group, controllerNode, graph = getGraph()) {
    const allNodes = getNodes(graph);
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
    for (const node of getNodes()) {
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

    async setup() {
        setInterval(refreshAllControllers, 1000);
    },
});
